"""Optional admission observations and documented reviews, without invented risk scores."""
# Limits are widget input limits, not laboratory reference intervals.
EXTRA_FIELDS = {
 'pain_score': ('Pain score (0–10)', r'pain(?: score)?', 0, 10, int),
 'gcs': ('Glasgow Coma Scale (3–15)', r'glasgow coma scale|gcs', 3, 15, int),
 'oxygen_flow': ('Oxygen flow (L/min)', r'oxygen flow(?: rate)?', 0, 100, float),
 'urine_volume': ('Urine volume (mL)', r'urine (?:output|volume)', 0, 20000, float),
 'urine_hours': ('Urine collection duration (hours)', r'urine (?:collection )?(?:duration|hours)', 0, 168, float),
 'sodium': ('Sodium (mmol/L)', r'sodium', 0, 250, float),
 'chloride': ('Chloride (mmol/L)', r'chloride', 0, 200, float),
 'bicarbonate': ('Bicarbonate / total CO2 (mmol/L)', r'bicarbonate|total co2', 0, 100, float),
 'calcium': ('Total calcium (mg/dL)', r'(?:total )?calcium', 0, 30, float),
 'magnesium': ('Magnesium (mg/dL)', r'magnesium', 0, 20, float),
 'phosphate': ('Phosphate (mg/dL)', r'phosphate|phosphorus', 0, 30, float),
 'albumin': ('Albumin (g/dL)', r'albumin', 0, 10, float),
 'bilirubin': ('Total bilirubin (mg/dL)', r'(?:total )?bilirubin', 0, 100, float),
 'alt': ('ALT (U/L)', r'alt|alanine aminotransferase', 0, 20000, float),
 'ast': ('AST (U/L)', r'ast|aspartate aminotransferase', 0, 20000, float),
 'alp': ('Alkaline phosphatase (U/L)', r'alkaline phosphatase|alp', 0, 10000, float),
 'hba1c': ('HbA1c (%)', r'hba1c|hemoglobin a1c|a1c', 0, 25, float),
}
ZERO_VALID={'pain_score','oxygen_flow','urine_volume'}
SCREENS={
 'vte_review':'VTE / clotting and bleeding review',
 'falls_review':'Falls and mobility review',
 'skin_review':'Skin and pressure-injury review',
 'nutrition_review':'Nutrition and swallowing review',
 'allergy_review':'Allergy and medication reconciliation review',
}
CHOICES=['Not assessed','Reviewed — concerns identified','Reviewed — no concerns identified','Not applicable']
REFERENCES={
 'Admission observations':'https://www.nice.org.uk/guidance/cg50/ifp/chapter/Arriving-on-the-ward-or-in-the-emergency-department',
 'Metabolic-panel tests':'https://medlineplus.gov/lab-tests/comprehensive-metabolic-panel-cmp/',
 'VTE review':'https://www.nice.org.uk/guidance/ng89/chapter/Recommendations',
 'Nutrition screening':'https://www.nice.org.uk/guidance/cg32/chapter/Recommendations',
 'Pressure-injury assessment':'https://www.nice.org.uk/guidance/cg179/chapter/Recommendations',
}

def render_inputs(prefix,observations=None,checks=None,on_change=None):
    import streamlit as st
    observations=observations or {}; checks=checks or {}
    callback={'on_change':on_change} if on_change else {}
    st.markdown('#### Additional admission observations and indicated labs')
    st.caption('Optional fields: blank means not measured. A recorded zero is valid for pain, oxygen flow, and urine volume. Enter tests only when clinically indicated; use the units shown and the reporting laboratory’s reference intervals.')
    values={}
    with st.expander('Pain, consciousness, oxygen, and urine output',expanded=True):
        columns=st.columns(3)
        for i,key in enumerate(list(EXTRA_FIELDS)[:5]):
            label,_,low,high,cast=EXTRA_FIELDS[key]
            value=observations.get(key)
            values[key]=columns[i%3].number_input(label,min_value=cast(low),max_value=cast(high),value=cast(value) if value is not None else None,key=prefix+key,**callback)
        options=['Unknown','Room air','Supplemental oxygen']
        values['oxygen_support']=st.selectbox('Oxygen support',options,index=options.index(observations.get('oxygen_support','Unknown')),key=prefix+'oxygen_support',**callback)
    with st.expander('Electrolytes, liver tests, and longer-term glucose control'):
        columns=st.columns(3)
        for i,key in enumerate(list(EXTRA_FIELDS)[5:]):
            label,_,low,high,cast=EXTRA_FIELDS[key]; value=observations.get(key)
            values[key]=columns[i%3].number_input(label,min_value=cast(low),max_value=cast(high),value=cast(value) if value is not None else None,key=prefix+key,**callback)
    with st.expander('Admission screening documentation'):
        st.caption('Record clinician review using your hospital’s tools. These selections are not validated risk scores and do not order treatment.')
        reviews={key:st.selectbox(label,CHOICES,index=CHOICES.index(checks.get(key,'Not assessed')),key=prefix+key,**callback) for key,label in SCREENS.items()}
        reviews['review_note']=st.text_area('Screening findings / reviewer / follow-up plan',value=checks.get('review_note',''),key=prefix+'review_note',**callback)
    return values,reviews

def summary(observations,checks=None):
    checks=checks or {}; rows=[]
    for key,label in [('o2_sat','Oxygen saturation (%)'),*[(k,v[0]) for k,v in EXTRA_FIELDS.items()]]:
        value=observations.get(key)
        missing=value is None or (value==0 and key not in ZERO_VALID)
        rows.append({'Assessment':label,'Result':'Not measured' if missing else f'{value:g}',
                     'Meaning':'No conclusion available' if missing else 'Recorded value — interpret with symptoms and clinical/lab context'})
    support=observations.get('oxygen_support','Unknown')
    rows.append({'Assessment':'Oxygen support','Result':support,'Meaning':'Confirm oxygen device and prescribed saturation target.'})
    volume=observations.get('urine_volume'); hours=observations.get('urine_hours')
    weight=observations.get('weight') or observations.get('weight_input')
    if observations.get('weight') is None and observations.get('w_unit')=='lbs' and weight: weight*=0.453592
    if volume is not None and hours and hours>0 and weight and weight>0:
        rows.append({'Assessment':'Urine output rate','Result':f'{volume/hours/weight:.2f} mL/kg/hour','Meaning':'Derived over the entered collection interval; not AKI staging.'})
    for key,label in SCREENS.items():
        rows.append({'Assessment':label,'Result':checks.get(key,'Not assessed'),'Meaning':'Documented clinician review, not an automatic risk calculation.'})
    return rows

def render_summary(observations,checks=None):
    import streamlit as st
    st.markdown('#### Broader admission review')
    st.dataframe(summary(observations,checks),hide_index=True,use_container_width=True)
    st.caption('Core observations and condition-specific tests serve different purposes. ECG, troponin, cultures, blood gases, and imaging require symptom-specific decisions; this app does not recommend them for every admission. qSOFA/SIRS do not rule out sepsis.')
    with st.expander('Clinical reference information'):
        for title,url in REFERENCES.items(): st.markdown(f'[{title}]({url})')
