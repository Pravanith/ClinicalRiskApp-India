"""Admission-to-discharge workspace with encounter-scoped state."""
import copy
import io
import json
import os
import uuid
from datetime import datetime, timezone
import streamlit as st
import care_store as store
import care_logic as logic
import care_ai
import admission
import india_documentation as india
from care_evidence import evidence_rows
from voice_entry import FIELDS
from voice_ui import speech_model


def setting(name):
    if os.environ.get(name): return os.environ[name]
    try: return st.secrets.get(name,'')
    except Exception: return ''


def reload_draft(eid):
    current=store.get_encounter(eid)
    st.session_state.pop('care_undo_'+eid,None)
    st.session_state.pop('care_extraction_'+eid,None)
    st.session_state['care_draft_'+eid]=copy.deepcopy(current['data'])
    st.session_state['care_version_'+eid]=current['version']
    st.session_state['care_epoch_'+eid]=st.session_state.get('care_epoch_'+eid,0)+1
    return current


def persist(eid,data,event,stage=None,reset_reviews=True):
    draft=copy.deepcopy(data)
    if reset_reviews: draft['discharge_checks']={}
    try:
        store.save(eid,draft,st.session_state['care_version_'+eid],stage,event)
        reload_draft(eid)
        st.session_state['care_flash']='Saved: '+event
        st.rerun()
    except ValueError as exc: st.error(str(exc))


def apply_note(eid,text):
    key='care_draft_'+eid
    st.session_state['care_undo_'+eid]=copy.deepcopy(st.session_state[key])
    draft,changes,warnings=logic.intake_from_text(text,st.session_state[key])
    draft['voice_evidence'] = evidence_rows(text, st.session_state[key], draft)
    st.session_state[key]=draft
    st.session_state['care_extraction_'+eid]=(changes,warnings)
    st.session_state['care_epoch_'+eid]=st.session_state.get('care_epoch_'+eid,0)+1


def recorded(eid,audio_key):
    recording=st.session_state.get(audio_key)
    if recording is None: return
    try:
        if recording.size>20*1024*1024: raise ValueError('Use a recording under 20 MB.')
        segments,_=speech_model().transcribe(io.BytesIO(recording.getvalue()),language='en',vad_filter=True,
             condition_on_previous_text=False,initial_prompt='Hospital admission, medical history, diabetes, hypertension, chronic kidney disease, allergies, metformin, apixaban, medications, dosage, creatinine, blood pressure, heart rate, hemoglobin, INR, eGFR, sodium, chloride, bicarbonate, calcium, magnesium, phosphate, albumin, bilirubin, ALT, AST, alkaline phosphatase, HbA1c, pain score, Glasgow Coma Scale, oxygen flow, urine volume, urine duration.')
        text=' '.join(s.text.strip() for s in segments).strip()
        if not text: raise ValueError('No speech detected. Please record again.')
        st.session_state['care_transcript_'+eid]=text
        apply_note(eid,text)
        st.session_state.pop('care_voice_error_'+eid,None)
    except Exception:
        st.session_state['care_voice_error_'+eid]='Audio could not be transcribed. Check microphone/audio, speech model availability, and recording length. Existing values are unchanged; typed notes remain available.'


def render_hospital(model):
    store.init()
    st.subheader('Hospital care workspace')
    st.caption('Registration → admission → ongoing review → discharge • Local prototype for clinician-reviewed documentation')
    st.warning('Use demonstration or appropriately authorized data. This local prototype has no user authentication, role controls, or production hospital integration.')
    people=store.patients()
    with st.expander('Register a new patient',expanded=not people):
        with st.form('care_registration'):
            a,b=st.columns(2)
            name=a.text_input('Patient display name')
            mrn=b.text_input('Patient reference / MRN')
            age=a.number_input('Age at registration',1,120,18)
            sex=b.selectbox('Sex for existing model',['Unknown','Male','Female'])
            if st.form_submit_button('Register patient'):
                try:
                    pid=store.register(mrn,name,age,sex)
                    st.session_state['care_patient']=pid
                    st.rerun()
                except ValueError as exc: st.error(str(exc))
    if not people: st.info('Register a patient to start an encounter.'); return
    by_id={p['id']:p for p in people}
    pid=st.selectbox('Patient',list(by_id),format_func=lambda p:by_id[p]['name']+' · '+by_id[p]['mrn'],key='care_patient')
    episodes=store.encounters(pid)
    with st.expander('Start a hospital encounter',expanded=not episodes):
        with st.form('care_new_encounter_'+pid):
            reason=st.text_input('Presenting complaint / admission reason')
            ward=st.text_input('Ward / service')
            if st.form_submit_button('Create encounter'):
                try:
                    eid=store.create_encounter(pid,reason,ward)
                    st.session_state['care_encounter_'+pid]=eid
                    st.rerun()
                except ValueError as exc: st.error(str(exc))
    if not episodes: return
    episode_map={r['id']:r for r in episodes}
    eid=st.selectbox('Encounter',list(episode_map),format_func=lambda e:episode_map[e]['admitted_at'][:16]+' · '+episode_map[e]['stage']+' · '+e[:8],key='care_encounter_'+pid)
    current=store.get_encounter(eid)
    if 'care_draft_'+eid not in st.session_state: reload_draft(eid)
    if st.button('Reload saved encounter / discard unsaved draft'):
        reload_draft(eid); st.rerun()
    if st.session_state.get('care_flash'): st.success(st.session_state.pop('care_flash'))
    if current['version']!=st.session_state['care_version_'+eid]:
        st.error('A newer version was saved in another session. Reload before editing.'); return
    st.markdown(f"**{current['name']} · {current['mrn']}** — {current['stage']}")
    st.caption('Encounter '+eid[:8]+' • Saved '+current['updated_at']+' • Times shown in UTC')
    if current['stage']=='Discharged':
        st.success('Discharged encounter — read-only. Create a new encounter for a subsequent admission.')
        summary=logic.discharge_text(current)
        st.markdown(summary)
        st.download_button('Download discharge summary',summary,file_name='discharge-'+eid[:8]+'.md')
        render_timeline(eid)
        return
    data=st.session_state['care_draft_'+eid]
    if data.get('previous_encounter'):
        st.info('Previous disease history, allergies, and discharge medications were copied as candidates. Reconfirm them for this admission; old vital signs and lab values were not copied.')
    if data!=current['data']: st.info('You have an unsaved transcript draft. Review and save a section to preserve it.')
    epoch=st.session_state.get('care_epoch_'+eid,0)
    prefix=eid+'_'+str(epoch)+'_'
    render_overview(data, current)
    tabs=st.tabs(['1 · Intake & voice','2 · Observations','3 · Risks','4 · Medications & AI','5 · Nutrition','6 · Discharge','Timeline'])
    with tabs[0]: render_intake(eid,data,prefix,current)
    with tabs[1]: render_observations(eid,data,prefix)
    with tabs[2]:
        st.markdown('#### Risk review from current encounter values')
        st.caption('Observation time: '+(data.get('observed_at') or 'Not recorded')+'. Draft edits are included; save reviewed observations to preserve them.')
        try:
            if not data.get('observed_at') or (datetime.now(timezone.utc)-datetime.fromisoformat(data['observed_at'])).total_seconds()>86400:
                st.warning('Observations are undated or more than 24 hours old. Confirm a current set before interpreting these results.')
        except ValueError: st.warning('Observation time needs review.')
        st.dataframe(logic.risk_review(data,model),hide_index=True,use_container_width=True)
        admission.render_summary(data.get('observations',{}),data.get('admission_checks',{}))
        st.write('Recorded previous conditions: '+(', '.join(data.get('conditions',[])) or 'None recorded — confirm history review.'))
        st.caption('The existing bleeding estimator uses six features. Disease and medication context also drives local checks and AI review; this is not a validated all-disease prediction model.')
    with tabs[3]: render_medications(eid,data,prefix)
    with tabs[4]: render_nutrition(eid,data,prefix)
    with tabs[5]: render_discharge(eid,data,prefix,current,model)
    with tabs[6]: render_timeline(eid)


def render_intake(eid,data,prefix,current):
    st.markdown('#### Dictate once, review the extracted details')
    st.caption('Press Stop to transcribe locally on this app server and fill the encounter draft. Nothing is saved to the encounter until you save a section. Silence alone does not stop recording.')
    audio_key='care_audio_'+eid
    st.audio_input('Admission or progress note',key=audio_key,on_change=recorded,args=(eid,audio_key))
    if st.session_state.get('care_voice_error_'+eid): st.error(st.session_state['care_voice_error_'+eid])
    left, right = st.columns(2)
    with left:
        text=st.text_area('Clinical transcript / typed note',key='care_transcript_'+eid,height=240)
        st.caption('Correct the note here, then fill the draft again. Unmentioned observations remain unchanged.')
    with right:
        st.markdown('**Source-linked changes from the last fill**')
        rows = data.get('voice_evidence', [])
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
            st.caption('Evidence links explain extraction, not clinical correctness. Review changed values and all surrounding context before saving. These rows describe the last voice fill; later manual edits may differ.')
        else:
            st.info('Record a note or paste text and fill the draft to see changes and supporting phrases here.')
    a,b=st.columns(2)
    if a.button('Fill encounter draft from note',key=prefix+'parse'):
        apply_note(eid,text); st.rerun()
    if b.button('Undo last transcript fill',key=prefix+'undo',disabled='care_undo_'+eid not in st.session_state):
        st.session_state['care_draft_'+eid]=st.session_state.pop('care_undo_'+eid)
        st.session_state['care_epoch_'+eid]+=1
        st.session_state.pop('care_extraction_'+eid,None); st.rerun()
    with st.expander('Example dictation and supported phrases'):
        st.code('Age 68. Blood pressure 120 over 80. Heart rate 76. Creatinine 1.2. eGFR 55.\nHistory of diabetes. No history of heart failure.\nAllergies: penicillin rash.\nTaking metformin 500 mg orally twice daily for type 2 diabetes.\nChief complaint: dizziness.')
        st.write('Say field names before values. Disease history requires “history of”, “has”, or an explicit denial. Medication candidates require “taking” or “medication”, a name, dose, and unit. Review formulation, route, indication, and frequency; uncertain fields remain unknown. Names and patient identifiers are entered at registration to prevent accidental reassignment.')
    extraction=st.session_state.get('care_extraction_'+eid)
    if extraction:
        st.info('Draft only: '+('; '.join(extraction[0]) or 'No supported structured fields found. Narrative retained.'))
        for warning in extraction[1]: st.warning(warning)
    with st.form(prefix+'intake'):
        reason=st.text_input('Presenting complaint',value=data.get('reason',''))
        ward=st.text_input('Ward / service',value=data.get('ward',''))
        conditions=st.multiselect('Previous / current diagnosed conditions',logic.CONDITIONS,default=data.get('conditions',[]))
        reviewed=st.checkbox('Disease history reviewed (including explicit absence)',value=data.get('conditions_reviewed',False))
        allergy_status=st.selectbox('Allergy history',['Unknown','No known allergies','Reported'],index=['Unknown','No known allergies','Reported'].index(data.get('allergy_status','Unknown')))
        allergies=st.text_area('Drug and food allergies, reactions',value=data.get('allergies',''))
        pregnancy=st.selectbox('Pregnancy / lactation context',['Unknown','Not applicable','Pregnant','Breastfeeding'],index=['Unknown','Not applicable','Pregnant','Breastfeeding'].index(data.get('pregnancy','Unknown')))
        history=st.text_area('Narrative history / current note',value=data.get('history_note',''))
        if st.form_submit_button('Save reviewed intake and transcript candidates'):
            if allergy_status=='Reported' and not allergies.strip(): st.error('Describe the reported allergies.'); return
            if allergy_status=='No known allergies' and allergies.strip(): st.error('Resolve the allergy text before marking no known allergies.'); return
            update={**data,'reason':reason,'ward':ward,'conditions':conditions,'conditions_reviewed':reviewed,'allergy_status':allergy_status,'allergies':allergies,'pregnancy':pregnancy,'history_note':history}
            # Extracted observations/medications are saved as candidates; their separate review remains required.
            persist(eid,update,'Intake reviewed')
    if current['stage']=='Registered' and st.button('Mark admitted',key=prefix+'admit',disabled=data!=current['data']):
        persist(eid,current['data'],'Admission confirmed',stage='Admitted')


def render_observations(eid,data,prefix):
    obs=data.get('observations',{})
    st.caption('Zeros represent missing values. Lab units follow the original calculator. Record a new set to build a timeline; past snapshots remain in the event history.')
    with st.form(prefix+'observations'):
        cols=st.columns(3); values={}
        for index,(key,(label,_,low,high,cast)) in enumerate(FIELDS.items()):
            value=obs.get(key,cast(0))
            units={'age':'years','height':'cm','sys_bp':'mmHg','dia_bp':'mmHg','hr':'beats/min','resp_rate':'breaths/min','o2_sat':'%','creat':'mg/dL','bun':'mg/dL','potassium':'mmol/L','glucose':'mg/dL','wbc':'10^9/L','hgb':'g/dL','platelets':'10^9/L','lactate':'mmol/L'}
            display=label+(' ('+units[key]+')' if key in units else '')
            if key == 'temp_c':
                fahrenheit = cols[index%3].number_input('Temperature °F', min_value=32.0, max_value=113.0, value=float(value)*9/5+32 if value else None, step=0.1, key=prefix+'obs_temp_f', placeholder='Not recorded')
                values[key] = (fahrenheit-32)*5/9 if fahrenheit is not None else 0.0
            else:
                values[key]=cols[index%3].number_input(display,min_value=cast(low),max_value=cast(high),value=cast(value),key=prefix+'obs_'+key)
        values['egfr']=st.number_input('eGFR (mL/min/1.73 m²)',0.0,200.0,float(obs.get('egfr',0)))
        values['gender']=st.selectbox('Sex for bleeding model',['Unknown','Male','Female'],index=['Unknown','Male','Female'].index(obs.get('gender','Unknown')))
        values['w_unit']=st.selectbox('Weight unit',['kg','lbs'],index=1 if obs.get('w_unit')=='lbs' else 0)
        mental=st.selectbox('Mental status',['Unknown','Normal','Altered'],index=['Unknown','Normal','Altered'].index(data.get('mental_status','Unknown')))
        additional, admission_checks = admission.render_inputs(prefix+'extra_',obs,data.get('admission_checks',{}))
        values.update(additional)
        measured=st.text_input('Measured at (ISO date/time with timezone)',value=data.get('observed_at') or store.now())
        if st.form_submit_button('Save reviewed observations'):
            if values['sys_bp'] and values['dia_bp']>=values['sys_bp']: st.error('Diastolic BP must be below systolic BP.'); return
            try:
                stamp=datetime.fromisoformat(measured)
                if stamp.tzinfo is None or stamp>datetime.now(timezone.utc): raise ValueError()
            except ValueError: st.error('Use a valid observation timestamp with timezone, not in the future.'); return
            persist(eid,{**data,'observations':values,'observed_at':stamp.isoformat(),'mental_status':mental,'admission_checks':admission_checks},'Observations reviewed')


def render_medications(eid,data,prefix):
    st.markdown('#### Medication reconciliation')
    st.caption('Record the actual reported regimen. Local rules cover a limited subset; AI and clinician review are needed for indication-specific doses and other medicines.')
    meds=copy.deepcopy(data.get('medications',[]))
    for index,med in enumerate(meds):
        with st.expander(med['name']+' · '+med.get('decision','Unreviewed')):
            with st.form(prefix+'med_'+med['id']):
                med['name']=st.text_input('Medication / generic name',value=med['name'])
                med['dose']=st.number_input('Amount per administration',0.0,100000.0,float(med.get('dose',0)))
                for field,label,options in [('unit','Dose unit',['mg','mcg','units','ml']),('frequency','Frequency',list(logic.FREQUENCIES)),('route','Route',['Unknown','Oral','IV','Subcutaneous','Inhaled','Topical','Other']),('formulation','Formulation',['Unknown','Immediate release','Extended release','Other']),('decision','Clinician reconciliation decision',['Unreviewed','Continue','Change','Stop','Start'])]:
                    med[field]=st.selectbox(label,options,index=options.index(med.get(field,options[0])))
                med['indication']=st.text_input('Indication',value=med.get('indication',''))
                if med.get('source'): st.caption('Entry source: '+med['source'])
                med['discharge_instruction']=st.text_area('Clinician-entered discharge instructions (dose, route, frequency, duration)',value=med.get('discharge_instruction',''))
                if st.form_submit_button('Save medication'):
                    if not med['name'].strip(): st.error('Medication name is required.'); return
                    updated=copy.deepcopy(data.get('medications',[])); updated[index]=med
                    persist(eid,{**data,'medications':updated,'medications_reviewed':False},'Medication reconciled')
            if st.button('Remove erroneous entry',key=prefix+'remove_'+med['id']):
                persist(eid,{**data,'medications':[m for m in data['medications'] if m['id']!=med['id']],'medications_reviewed':False},'Medication entry removed')
    with st.form(prefix+'add_med'):
        name=st.text_input('Add reported medication by name')
        if st.form_submit_button('Add medication details'):
            if name.strip():
                new={'id':uuid.uuid4().hex,'name':name.strip(),'dose':0.0,'unit':'mg','frequency':'Unknown','route':'Unknown','formulation':'Unknown','indication':'','decision':'Unreviewed','discharge_instruction':''}
                persist(eid,{**data,'medications':data.get('medications',[])+[new],'medications_reviewed':False},'Medication added')
            else: st.error('Enter a medication name.')
    if st.button('Confirm medication history reviewed (including no medications)',key=prefix+'med_reviewed'):
        persist(eid,{**data,'medications_reviewed':True},'Medication history reviewed')
    st.caption('Medication history reviewed: '+str(data.get('medications_reviewed',False)))
    st.dataframe(logic.medication_review(data),hide_index=True,use_container_width=True)
    st.caption('“Dose not verified” and “no issue identified” are not approval to prescribe or administer. Full renal/hepatic adjustment, class allergies, PRN limits, and indication-specific regimens may require additional information.')
    render_ai(eid,data,prefix)


def render_ai(eid,data,prefix):
    st.markdown('#### AI cross-check and care discussion')
    with st.expander('AI settings'):
        st.text_input('Gemini API key (session only)',type='password',key='hospital_ai_key')
        st.text_input('Gemini model ID',key='hospital_ai_model',placeholder='Use an available model from your Gemini account')
        st.caption('Alternatively configure GEMINI_API_KEY and GEMINI_MODEL in Streamlit secrets or environment variables.')
    payload=care_ai.clinical_payload(data)
    with st.expander('Preview clinical details sent to Gemini'):
        st.json(payload)
    st.caption('The button below sends these clinical details to Google Gemini and medication names to openFDA for candidate product labels. Identity fields and the raw transcript are excluded; review free-text allergy and indication fields for identifiers.')
    if st.button('Send shown details for AI review',key=prefix+'ai'):
        try:
            with st.spinner('Checking product labels and requesting clinical review…'):
                report=care_ai.ai_review(data,st.session_state.get('hospital_ai_key') or setting('GEMINI_API_KEY'),st.session_state.get('hospital_ai_model') or setting('GEMINI_MODEL'))
            report['created_at']=store.now()
            persist(eid,{**data,'ai_review':report},'AI draft review generated',reset_reviews=False)
        except care_ai.ReviewError as exc: st.error(str(exc))
    report=data.get('ai_review')
    if report:
        if report.get('payload')!=payload or report.get('basis_observed_at')!=data.get('observed_at'):
            st.warning('The previous AI review is out of date after clinical changes. Request a new review.'); return
        st.warning('AI-generated draft — clinician verification required. It does not replace product labeling or a pharmacist.')
        answer=report['answer']; st.write(answer['summary'])
        ids={m['id']:m['name'] for m in data.get('medications',[])}
        for row in answer['medication_reviews']:
            with st.container(border=True):
                st.markdown('**'+ids[row['medication_id']]+' — '+row['status']+'**')
                st.write(row['reason']); st.write('For clinician review: '+row['clinician_action'])
                if row['missing_information']: st.write('Missing: '+', '.join(row['missing_information']))
        for title,key in [('Risk considerations','risk_considerations'),('Treatment options to discuss','treatment_discussion'),('Nutrition considerations','nutrition_considerations')]:
            st.markdown('**'+title+'**')
            for item in answer[key]: st.write('• '+item)
        with st.expander('Product-label evidence and review provenance'):
            st.write('Model: '+report['model']+' • '+report['created_at'])
            st.json(report['evidence'])


def render_nutrition(eid,data,prefix):
    with st.form(prefix+'nutrition'):
        options=['Unknown','Cleared for oral intake','NPO / nothing by mouth','Swallowing review required']
        oral=st.selectbox('Current oral intake clearance',options,index=options.index(data.get('oral_intake','Unknown')))
        diets=['Standard','Vegetarian','Vegan']
        diet=st.selectbox('Dietary preference',diets,index=diets.index(data.get('diet','Standard')))
        note=st.text_area('Clinician / dietitian instructions and prescribed restrictions',value=data.get('nutrition_note',''))
        if st.form_submit_button('Save nutrition context'):
            persist(eid,{**data,'oral_intake':oral,'diet':diet,'nutrition_note':note},'Nutrition context reviewed')
    plan=logic.nutrition_plan(data)
    st.markdown('#### '+plan['status'])
    for tip in plan['advice']: st.write('• '+tip)
    if plan['meals']: st.dataframe(plan['meals'],hide_index=True,use_container_width=True)
    for source in plan['sources']: st.markdown('[Nutrition reference]('+source+')')
    st.caption('Meal suggestions are drafts, not hospital diet orders. The saved clinician/dietitian instructions are included in discharge documentation.')


def render_discharge(eid,data,prefix,current,model):
    st.info('India deployment: US openFDA labels and the existing dose rules do not verify Indian products. A registered clinician/pharmacist must check the actual Indian product, ingredient, strength, route, and formulation.')
    st.caption('Prepare a draft at any stage. Finalization requires admission and documented clinician review. The app does not determine medical readiness for discharge.')
    st.dataframe(logic.risk_review(data,model),hide_index=True,use_container_width=True)
    with st.form(prefix+'discharge'):
        updated=copy.deepcopy(data)
        updated['india_documentation_enabled'] = st.checkbox('Apply India documentation checklist to this encounter', value=data.get('india_documentation_enabled', False))
        st.caption('Documentation prompts informed by CEA hospital standards. Applicability depends on state and facility. Save the draft to update finalization checks. Contact is optional; no ABHA or Aadhaar is required by this app.')
        for field, label in india.FIELDS.items():
            updated[field] = st.text_area(label, value=data.get(field, 'Telangana' if field == 'facility_state' else ''), help='Target locations: Warangal and Karimnagar. Enter the actual facility location.' if field == 'facility_city' else None)
        for key,label in [('diagnosis','Final diagnosis'),('course','Hospital course and procedures'),('followup','Follow-up appointments, pending tests, and responsible service'),('return_precautions','Return precautions / whom to contact'),('clinician','Reviewing clinician name')]:
            updated[key]=st.text_area(label,value=data.get(key,''))
        updated['discharge_checks']={k:st.checkbox(label,value=data.get('discharge_checks',{}).get(k,False)) for k,label in [('medications','Medication reconciliation and written instructions reviewed'),('risks','Current risks, unresolved issues, and discharge readiness reviewed'),('nutrition','Nutrition plan and restrictions reviewed'),('instructions','Patient instructions and follow-up reviewed')]}
        if st.form_submit_button('Save discharge draft'):
            persist(eid,updated,'Discharge draft saved',reset_reviews=False)
    missing=store.discharge_missing(data)
    if missing: st.info('Before finalization: '+', '.join(missing))
    if st.button('Finalize clinician-reviewed discharge',key=prefix+'discharge_final',disabled=bool(missing) or current['stage']!='Admitted'):
        persist(eid,data,'Discharge finalized',stage='Discharged',reset_reviews=False)
    draft={**current,'data':data}
    st.download_button('Download draft discharge summary',logic.discharge_text(draft),file_name='discharge-draft-'+eid[:8]+'.md',key=prefix+'download')


def render_timeline(eid):
    rows=store.events(eid)
    st.markdown('#### Encounter timeline')
    st.dataframe([{'Time (UTC)':r['created_at'],'Event':r['event']} for r in rows],hide_index=True,use_container_width=True)
    observations=[]
    for row in rows:
        if row['event']=='Observations reviewed':
            d=json.loads(row['data']); observations.append({'Measured at':d.get('observed_at'),**d.get('observations',{})})
    if observations:
        st.markdown('**Observation history**')
        st.dataframe(observations,hide_index=True,use_container_width=True)


def render_overview(data, current):
    with st.expander('Patient overview · current encounter', expanded=True):
        a, b, c = st.columns(3)
        a.metric('Encounter', current['stage'])
        b.metric('Recorded medications', len(data.get('medications', [])))
        c.metric('Observation fields', len(data.get('observations', {})))
        st.write('**Presenting complaint:**', data.get('reason') or 'Not recorded')
        st.write('**Conditions:**', ', '.join(data.get('conditions', [])) or 'None recorded; confirm history')
        st.write('**Allergies:**', data.get('allergy_status', 'Unknown'), '—', data.get('allergies') or 'No reaction details recorded')
        pending = []
        if not data.get('conditions_reviewed'): pending.append('Disease history review')
        if not data.get('medications_reviewed'): pending.append('Medication history review')
        if data.get('allergy_status', 'Unknown') == 'Unknown': pending.append('Allergy history')
        if not data.get('observed_at'): pending.append('Reviewed observation time')
        st.write('**Pending review:**', ', '.join(pending) or 'These intake checks are recorded; this is not discharge clearance.')
        st.caption('Current draft is shown. Counts indicate recorded entries, not completeness. Use the sections below to review and save.')
