"""India-oriented documentation prompts; not accreditation or clinical rules."""
from datetime import datetime
from zoneinfo import ZoneInfo

FIELDS = {
 'facility_name': 'Hospital name',
 'facility_city': 'Hospital city / district',
 'facility_state': 'Hospital state / Union Territory',
 'facility_contact': 'Hospital contact / urgent-care contact',
 'doctor_registration': 'Treating doctor registration number and medical council',
 'patient_contact': 'Patient contact, if available (fictional details in demo)',
 'assessment_summary': 'Clinical history and assessment findings',
 'investigation_summary': 'Investigation results / report references (or explicitly none)',
 'discharge_condition': 'Condition at discharge',
 'instruction_language': 'Language used for patient instructions',
 'patient_instructions': 'Patient-facing instructions in the chosen language',
 'consent_reference': 'Applicable consent record references / not applicable with reason',
}
REQUIRED = ('facility_name','facility_state','facility_contact','doctor_registration',
            'assessment_summary','investigation_summary','discharge_condition',
            'instruction_language','patient_instructions','consent_reference')

def missing(data):
    if not data.get('india_documentation_enabled'): return []
    return [FIELDS[k] for k in REQUIRED if not str(data.get(k,'')).strip()]

def ist(stamp):
    if not stamp: return 'Not recorded'
    try:
        dt = datetime.fromisoformat(stamp)
        if dt.tzinfo is None: return stamp+' (timezone unknown)'
        return dt.astimezone(ZoneInfo('Asia/Kolkata')).strftime('%d-%m-%Y %H:%M IST')
    except (ValueError, TypeError): return 'Invalid timestamp — review source'

def summary(encounter):
    data=encounter['data']
    if not data.get('india_documentation_enabled'): return []
    lines=['## India documentation supplement',
           'Admission: '+ist(encounter.get('admitted_at')),
           'Discharge: '+ist(encounter.get('discharged_at')),
           'Age: '+str(data.get('observations',{}).get('age','Not recorded')),
           'Recorded sex: '+str(data.get('observations',{}).get('gender','Unknown'))]
    for key,label in FIELDS.items():
        lines.extend(['### '+label, data.get(key) or 'Not recorded'])
    lines.append('Clinician identity and consent references are entered text, not authenticated signatures or verified consent. This supplement is not a compliance certificate.')
    return lines
