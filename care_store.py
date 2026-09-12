"""Patient/encounter storage isolated from the legacy demonstration history."""
import json
import copy
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / 'clinical_data.db'

def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

def connect():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys=ON')
    return conn

def init():
    with connect() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS care_patients (
          id TEXT PRIMARY KEY, mrn TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
          age INTEGER NOT NULL, sex TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS care_encounters (
          id TEXT PRIMARY KEY, patient_id TEXT NOT NULL REFERENCES care_patients(id),
          stage TEXT NOT NULL, data TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1,
          admitted_at TEXT NOT NULL, updated_at TEXT NOT NULL, discharged_at TEXT);
        CREATE UNIQUE INDEX IF NOT EXISTS care_one_open_encounter ON care_encounters(patient_id) WHERE stage != 'Discharged';
        CREATE TABLE IF NOT EXISTS care_events (
          id TEXT PRIMARY KEY, encounter_id TEXT NOT NULL REFERENCES care_encounters(id),
          event TEXT NOT NULL, data TEXT NOT NULL, created_at TEXT NOT NULL);
        ''')

def patients():
    with connect() as c: return [dict(r) for r in c.execute('SELECT * FROM care_patients ORDER BY created_at DESC')]

def register(mrn, name, age, sex):
    if not mrn.strip() or not name.strip(): raise ValueError('Patient reference and name are required.')
    if not 0 < age <= 120: raise ValueError('Enter an age between 1 and 120.')
    pid = uuid.uuid4().hex
    try:
        with connect() as c:
            c.execute('INSERT INTO care_patients VALUES(?,?,?,?,?,?)',(pid,mrn.strip().upper(),name.strip(),age,sex,now()))
    except sqlite3.IntegrityError as exc: raise ValueError('That patient reference already exists. Select the existing patient.') from exc
    return pid

def encounters(pid):
    with connect() as c: return [dict(r) for r in c.execute('SELECT * FROM care_encounters WHERE patient_id=? ORDER BY admitted_at DESC',(pid,))]

def create_encounter(pid, reason, ward):
    if not reason.strip(): raise ValueError('Enter a presenting complaint or admission reason.')
    with connect() as c:
        c.execute('BEGIN IMMEDIATE')
        patient=c.execute('SELECT * FROM care_patients WHERE id=?',(pid,)).fetchone()
        if not patient: raise ValueError('Patient not found.')
        if c.execute("SELECT 1 FROM care_encounters WHERE patient_id=? AND stage!='Discharged'",(pid,)).fetchone():
            raise ValueError('This patient already has an open encounter. Select it below.')
        eid=uuid.uuid4().hex
        data={'reason':reason.strip(),'ward':ward.strip(),'observations':{'age':patient['age'],'gender':patient['sex']},
              'conditions':[], 'conditions_reviewed':False, 'allergies':'', 'allergy_status':'Unknown',
              'medications':[], 'history_note':'', 'diet':'Standard', 'oral_intake':'Unknown',
              'observed_at':'', 'pregnancy':'Unknown', 'diagnosis':'', 'course':'', 'followup':'', 'return_precautions':'',
              'clinician':'', 'nutrition_note':'', 'discharge_checks':{}}
        previous=c.execute('SELECT id,data FROM care_encounters WHERE patient_id=? ORDER BY admitted_at DESC,rowid DESC LIMIT 1',(pid,)).fetchone()
        if previous:
            prior=json.loads(previous['data'])
            data['previous_encounter']=previous['id']
            data['conditions']=prior.get('conditions',[])
            data['allergies']=prior.get('allergies','')
            # Prior allergy and medication context is a candidate, never a current confirmation.
            data['allergy_status']='Reported' if data['allergies'] else 'Unknown'
            for medication in prior.get('medications',[]):
                if medication.get('decision') in ('Continue','Start','Change'):
                    candidate=copy.deepcopy(medication)
                    candidate['id']=uuid.uuid4().hex
                    candidate['source']='Previous discharge: '+medication.get('discharge_instruction','')
                    if candidate['decision']=='Change':
                        candidate['dose']=0.0; candidate['frequency']='Unknown'
                    candidate['decision']='Unreviewed'; candidate['discharge_instruction']=''
                    data['medications'].append(candidate)
        stamp=now()
        c.execute('INSERT INTO care_encounters VALUES(?,?,?,?,?,?,?,?)',(eid,pid,'Registered',json.dumps(data),1,stamp,stamp,None))
        c.execute('INSERT INTO care_events VALUES(?,?,?,?,?)',(uuid.uuid4().hex,eid,'Registered',json.dumps(data),stamp))
    return eid

def get_encounter(eid):
    with connect() as c:
        row=c.execute('SELECT e.*,p.name,p.mrn FROM care_encounters e JOIN care_patients p ON p.id=e.patient_id WHERE e.id=?',(eid,)).fetchone()
    if not row: raise ValueError('Encounter not found.')
    result=dict(row); result['data']=json.loads(result['data']); return result

def save(eid, data, version, stage=None, event='Clinical update'):
    with connect() as c:
        c.execute('BEGIN IMMEDIATE')
        row=c.execute('SELECT * FROM care_encounters WHERE id=?',(eid,)).fetchone()
        if not row or row['version'] != version: raise ValueError('This encounter changed elsewhere. Reload it before saving.')
        if row['stage']=='Discharged': raise ValueError('Discharged encounters are read-only. Open a new encounter for readmission.')
        target=stage or row['stage']
        allowed={'Registered':{'Registered','Admitted'},'Admitted':{'Admitted','Discharged'}}
        if target not in allowed.get(row['stage'],set()): raise ValueError('Complete admission before discharge.')
        if target=='Discharged':
            problems=discharge_missing(data)
            if problems: raise ValueError('Discharge incomplete: '+', '.join(problems))
        stamp=now()
        c.execute('UPDATE care_encounters SET data=?,stage=?,version=version+1,updated_at=?,discharged_at=? WHERE id=?',
                  (json.dumps(data,allow_nan=False),target,stamp,stamp if target=='Discharged' else None,eid))
        c.execute('INSERT INTO care_events VALUES(?,?,?,?,?)',(uuid.uuid4().hex,eid,event,json.dumps(data,allow_nan=False),stamp))
    return get_encounter(eid)

def events(eid):
    with connect() as c: return [dict(r) for r in c.execute('SELECT * FROM care_events WHERE encounter_id=? ORDER BY created_at,rowid',(eid,))]

def discharge_missing(data):
    missing=[label for key,label in [('diagnosis','diagnosis'),('course','hospital course'),('followup','follow-up'),('return_precautions','return precautions'),('clinician','reviewing clinician')] if not data.get(key,'').strip()]
    for key in ('medications','risks','nutrition','instructions'):
        if not data.get('discharge_checks',{}).get(key): missing.append(key+' review')
    if any(m.get('decision','Unreviewed')=='Unreviewed' for m in data.get('medications',[])): missing.append('medication decisions')
    if any(m.get('decision') in ('Continue','Change','Start') and not m.get('discharge_instruction','').strip() for m in data.get('medications',[])): missing.append('discharge medication instructions')
    from india_documentation import missing as india_missing
    return missing + india_missing(data)
