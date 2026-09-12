"""Reviewable intake extraction and evidence-linked decision support, not orders."""
import copy
import re
import uuid
from voice_entry import extract_fields, normalize_numbers

CONDITIONS=['Diabetes','Hypertension','Chronic kidney disease','Heart failure','Liver disease','Prior GI bleeding','Atrial fibrillation','Asthma/COPD','Cancer']
CONDITION_PATTERNS={'Diabetes':r'diabetes|diabetic','Hypertension':r'hypertension|high blood pressure',
 'Chronic kidney disease':r'chronic kidney disease|ckd','Heart failure':r'heart failure',
 'Liver disease':r'liver disease','Prior GI bleeding':r'(?:prior |history of )?gi bleed(?:ing)?',
 'Atrial fibrillation':r'atrial fibrillation|afib','Asthma/COPD':r'asthma|copd','Cancer':r'cancer'}
ALIASES={'eliquis':'apixaban','coumadin':'warfarin','lasix':'furosemide','glucophage':'metformin','advil':'ibuprofen','prinivil':'lisinopril'}
FREQUENCIES={'Unknown':None,'Once daily':1,'Twice daily':2,'Three times daily':3,'Four times daily':4,'Weekly':None,'As needed':None}
METFORMIN_SOURCE='https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=54bb8030-8e80-4b38-8deb-89c99d73bf09'
KIDNEY_SOURCE='https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/healthy-eating-adults-chronic-kidney-disease'
DASH_SOURCE='https://www.nhlbi.nih.gov/health/dash-eating-plan'

def drug_name(name): return ALIASES.get(name.lower().strip(),name.lower().strip())

def intake_from_text(text,data):
    result=extract_fields(text)
    draft=copy.deepcopy(data)
    draft.setdefault('observations',{}).update(result.values)
    changes=[f'{k}: {v}' for k,v in result.values.items()]
    # Unknown history remains unknown; narration never marks the whole history reviewed.
    conditions=set(draft.get('conditions',[]))
    for condition,pattern in CONDITION_PATTERNS.items():
        yes=bool(re.search(r'\b(?:history of|has|known)\s+(?:'+pattern+r')\b',text,re.I))
        no=bool(re.search(r'\b(?:no(?: history of)?|denies|without)\s+(?:'+pattern+r')\b',text,re.I))
        # The positive phrase "history of" inside a negation is not positive evidence.
        positive_text=re.sub(r'\b(?:no(?: history of)?|denies|without)\s+(?:'+pattern+r')\b','',text,flags=re.I)
        yes=bool(re.search(r'\b(?:history of|has|known)\s+(?:'+pattern+r')\b',positive_text,re.I))
        if yes and no: result.warnings.append(f'Conflicting {condition} history; unchanged.')
        elif yes: conditions.add(condition); changes.append(condition+' reported')
        elif no: conditions.discard(condition); changes.append(condition+' denied')
    draft['conditions']=[c for c in CONDITIONS if c in conditions]
    normalized=normalize_numbers(text)
    # eGFR requires its own label; never substitute creatinine or assume a renal value.
    matches=[]
    for match in re.finditer(r'\begfr\s*(?:is |of |:)?\s*(\d+(?:\.\d+)?)',normalized):
        prefix=normalized[max(0,match.start()-35):match.start()]
        tail=normalized[match.end():]
        if re.search(r'\b(previous|prior|baseline|yesterday|not|no)\s*$',prefix) or re.match(r'\s*(?:to|or|[-–])\s*\d',tail):
            result.warnings.append('Historical, negated, or ranged eGFR was not applied.'); continue
        matches.append(match.group(1))
    if len(set(matches))==1 and 0 < float(matches[0]) <= 200:
        draft['observations']['egfr']=float(matches[0]); changes.append('eGFR: '+matches[0])
    elif matches: result.warnings.append('eGFR needs manual review.')
    for label,key in [('chief complaint','reason'),('presenting complaint','reason'),('allergies','allergies')]:
        match=re.search(r'\b'+label+r'\s*:\s*([^\n;.]+)',text,re.I)
        if match:
            draft[key]=match.group(1).strip(); changes.append(label+' captured')
            if key=='allergies': draft['allergy_status']='Reported'
    # Intentionally bounded grammar: every medication remains an unreviewed candidate.
    medications=copy.deepcopy(draft.get('medications',[]))
    pattern=r'\b(?:medication|taking)\s+([a-z][a-z -]{1,45}?)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|units|ml|milligrams?|micrograms?|milliliters?)\b([^;\n]*?)(?=\.(?:\s|$)|;|\n|$)'
    for m in re.finditer(pattern,normalized):
        if re.search(r'\b(not|no longer|previously|stopped)\s*$',normalized[max(0,m.start()-35):m.start()]):
            result.warnings.append('Negated or historical medication mention was not added.'); continue
        name,dose,unit,tail=m.groups()
        unit={'milligram':'mg','milligrams':'mg','microgram':'mcg','micrograms':'mcg','milliliter':'ml','milliliters':'ml'}.get(unit,unit)
        formulation='Unknown'
        if re.search(r'extended release|\ber\b|\bxr\b',name+' '+tail): formulation='Extended release'
        elif re.search(r'immediate release',name+' '+tail): formulation='Immediate release'
        name=re.sub(r'\s+(?:extended release|immediate release|er|xr)$','',name).strip()
        frequency='Unknown'
        for pat,freq in [(r'once (?:a day|daily|per day)|daily|every day|\bqd\b','Once daily'),(r'twice (?:a day|daily|per day)|two times daily|2 times daily|every 12 hours|\bbid\b','Twice daily'),(r'three times daily|3 times daily|every 8 hours|\btid\b','Three times daily'),(r'four times daily|4 times daily|every 6 hours|\bqid\b','Four times daily'),(r'weekly','Weekly'),(r'as needed|prn','As needed')]:
            if re.search(pat,tail): frequency=freq
        indication=re.search(r'\bfor\s+(.+)',tail)
        med={'id':uuid.uuid4().hex,'name':name.strip(),'dose':float(dose),'unit':unit,
             'frequency':frequency,'route':'Oral' if re.search(r'oral|by mouth',tail) else 'Unknown',
             'formulation':formulation,
             'indication':indication.group(1).strip() if indication else '', 'decision':'Unreviewed',
             'discharge_instruction':'','source':m.group()}
        if any(drug_name(x['name'])==drug_name(med['name']) for x in medications):
            result.warnings.append(f"{name} already exists; reconcile manually instead of adding a duplicate."); continue
        medications.append(med); changes.append('Medication candidate: '+m.group())
    draft['medications']=medications
    draft['history_note']=text
    draft['discharge_checks']={}
    draft['conditions_reviewed']=False
    draft['medications_reviewed']=False
    if result.values or matches: draft['observed_at']=''
    if 'altered_mental' in result.values: draft['mental_status']='Altered' if result.values['altered_mental'] else 'Normal'
    return draft,changes,result.warnings

def medication_review(data):
    from drug_data import INTERACTION_DB
    findings=[]
    def flag(status,drug,message,source=''):
        findings.append({'Status':status,'Medication':drug,'Finding':message,'Source':source})
    meds=data.get('medications',[]); obs=data.get('observations',{}); conditions=data.get('conditions',[])
    names=[drug_name(m.get('name','')) for m in meds]
    for med,name in zip(meds,names):
        missing=[k for k in ('indication','route','formulation','frequency') if not med.get(k) or med[k]=='Unknown']
        if med.get('dose',0)<=0: missing.append('dose')
        if data.get('allergy_status','Unknown')=='Unknown': missing.append('allergy review')
        if not data.get('conditions_reviewed'): missing.append('medical history review')
        if data.get('pregnancy','Unknown')=='Unknown': missing.append('pregnancy/lactation applicability')
        if missing: flag('Insufficient information',name,'Confirm '+', '.join(missing)+'.')
        if data.get('allergy_status')=='Reported':
            allergy=data.get('allergies','').lower()
            if any(re.search(r'\b'+re.escape(token)+r'\b',allergy) for token in [name,med.get('name','').lower()] if token):
                flag('Potential allergy conflict',name,'Medication name matches reported allergy text. Verify reaction and allergy class; do not administer based on this app.')
            else: flag('Review required',name,'Reported allergies require manual ingredient/class cross-check; absence of an exact name match does not clear allergy risk.')
        if names.count(name)>1: flag('Possible duplication',name,'More than one entry has this active ingredient. Confirm intended regimen.')
        if name=='metformin':
            egfr=obs.get('egfr',0)
            if not egfr: flag('Insufficient information',name,'Current eGFR is needed for renal label checks.',METFORMIN_SOURCE)
            elif egfr<30: flag('Label conflict',name,'Recorded eGFR is below the label contraindication threshold of 30 mL/min/1.73 m². Prompt prescriber/pharmacist review.',METFORMIN_SOURCE)
            elif egfr<45: flag('Review required',name,'Initiation is not recommended at eGFR 30–44; ongoing treatment requires benefit/risk review.',METFORMIN_SOURCE)
            times=FREQUENCIES.get(med.get('frequency'))
            if obs.get('age',0)>=18 and med.get('route')=='Oral' and med.get('unit')=='mg' and times and med.get('formulation')=='Immediate release':
                daily=med.get('dose',0)*times
                if daily>2550: flag('Label conflict',name,f'Recorded total {daily:g} mg/day exceeds the cited adult immediate-release maximum of 2550 mg/day.',METFORMIN_SOURCE)
                else: flag('Limited check only',name,'The recorded daily amount does not exceed the cited adult immediate-release maximum. This does not establish a suitable starting dose or regimen.',METFORMIN_SOURCE)
            else: flag('Insufficient information',name,'Daily-dose rule covers adult oral immediate-release metformin in mg with a fixed daily schedule only.',METFORMIN_SOURCE)
        else: flag('Dose not verified',name,'No local dose rule covers this medicine. Check the exact product label and obtain AI/pharmacist review.')
    for i,name in enumerate(names):
        for other in names[i+1:]:
            rule=INTERACTION_DB.get((name,other)) or INTERACTION_DB.get((other,name))
            if rule: flag('Potential interaction',name+' + '+other,'Existing interaction library flags this pair. Confirm mechanism, severity, and management with current product labels.','Local interaction library; not a complete interaction database')
    if not meds: flag('Not assessed','Medication list','No medications recorded; confirm whether this means none or history is incomplete.')
    return findings

def risk_review(data,model=None):
    import backend as bk
    obs=data.get('observations',{}); conditions=data.get('conditions',[])
    values=[]; missing=[]
    def add(name,value,meaning): values.append({'Assessment':name,'Result':value,'Interpretation':meaning})
    if obs.get('age',0)<18:
        return [{'Assessment':'Adult scope','Result':'Not assessed','Interpretation':'Pediatric review is required.'}]
    if all(obs.get(k,0)>0 for k in ('sys_bp','resp_rate')) and data.get('mental_status','Unknown')!='Unknown':
        q=bk.calculate_sepsis_risk(obs['sys_bp'],obs['resp_rate'],data['mental_status']=='Altered',obs.get('temp_c',0))
        add('qSOFA',f'{q}/3','Clinical review flag; not a sepsis diagnosis. Interpret in suspected infection.')
    else: add('qSOFA','Incomplete','Systolic BP, respiratory rate, and reviewed mental status required.')
    if all(obs.get(k,0)>0 for k in ('sys_bp','dia_bp')):
        add('Mean arterial pressure',f"{(obs['sys_bp']+2*obs['dia_bp'])/3:.1f} mmHg",'Derived from entered blood pressure.')
    if obs.get('glucose',0)>0:
        g=obs['glucose']; add('Glucose',f'{g:g} mg/dL','Low reading: prompt clinical review.' if g<70 else 'Above 180: review timing and clinical context.' if g>180 else 'Interpret with timing, symptoms, and diabetes history.')
    drugs={drug_name(m['name']) for m in data.get('medications',[])}
    # Medication history drives context, but an unreviewed list cannot imply absence.
    if all(obs.get(k,0)>0 for k in ('age','sys_bp','creat')) and data.get('conditions_reviewed') and data.get('medications_reviewed'):
        aki=bk.calculate_aki_risk(obs['age'],bool(drugs & {'furosemide','hydrochlorothiazide','spironolactone'}),bool(drugs & {'lisinopril','losartan','valsartan'}),obs['sys_bp'],'Cancer' in conditions,obs['creat'],bool(drugs & {'ibuprofen','naproxen'}),'Heart failure' in conditions)
        add('Legacy AKI heuristic',f'{aki}/100','Limited drug mapping and local rules; not AKI staging or a clinical probability.')
    else: add('Legacy AKI heuristic','Incomplete','Age, systolic BP, creatinine, reviewed disease and medication histories required.')
    if model and all(obs.get(k,0)>0 for k in ('age','sys_bp','inr')) and obs.get('gender') in ('Male','Female') and data.get('conditions_reviewed') and data.get('medications_reviewed'):
        import pandas as pd
        anticoag=bool(drugs & {'apixaban','rivaroxaban','warfarin','dabigatran','edoxaban','heparin','enoxaparin'})
        frame=pd.DataFrame([{'age':obs['age'],'inr':obs['inr'],'systolic_bp':obs['sys_bp'],'gender':obs['gender'],'anticoagulant':int(anticoag),'liver_disease':int('Liver disease' in conditions)}])
        if isinstance(model,bk.HeuristicFallbackModel): add('Bleeding model','Unavailable','Original fallback is not used to estimate hospital encounter risk.')
        else:
            try: add('Bleeding model',f'{model.predict_proba(frame)[0][1]*100:.1f}%','Synthetic-data estimate; limited medication-class mapping. Not clinically validated.')
            except Exception: add('Bleeding model','Unavailable','Input/model compatibility needs review.')
    else: add('Bleeding model','Incomplete','Age, systolic BP, INR, model-compatible sex, reviewed disease and medication histories required.')
    return values

def nutrition_plan(data):
    conditions=set(data.get('conditions',[])); obs=data.get('observations',{})
    cautions=[]
    if data.get('oral_intake','Unknown')!='Cleared for oral intake':
        return {'status':'Oral meal suggestions withheld','advice':['Confirm swallowing safety, procedure restrictions, and the current diet order first.'],'meals':[],'sources':[]}
    if obs.get('age',0)<18 or data.get('allergy_status','Unknown')!='No known allergies' or data.get('pregnancy','Unknown')!='Not applicable':
        return {'status':'Individual nutrition review needed','advice':['Confirm age-specific needs, food allergies, and pregnancy/lactation requirements before selecting foods.'],'meals':[],'sources':[]}
    if not data.get('conditions_reviewed'):
        return {'status':'History review needed','advice':['Review prior conditions before generating a condition-aware meal draft.'],'meals':[],'sources':[]}
    if 'Chronic kidney disease' in conditions or 'Heart failure' in conditions or 'Liver disease' in conditions:
        return {'status':'Dietitian-led meal plan needed','advice':['Use the current prescribed diet. Individualize fluid, protein, potassium, phosphorus, and sodium targets with the treating team; a generic menu may conflict with these needs.'],'meals':[],'sources':[KIDNEY_SOURCE]}
    if data.get('nutrition_note','').strip():
        return {'status':'Recorded nutrition instructions','advice':[data['nutrition_note'], 'Review recorded restrictions with the dietitian before adding a menu.'],'meals':[],'sources':[]}
    advice=['Choose minimally processed foods, vegetables, suitable protein, and unsweetened drinks. Portions require individual assessment.']
    if 'Diabetes' in conditions: advice.append('Distribute carbohydrate-containing foods across meals; coordinate meal timing with the prescribed glucose-lowering regimen.')
    if 'Hypertension' in conditions: advice.append('Favor lower-sodium ingredients; use the clinician’s sodium target.')
    vegetarian=data.get('diet') in ('Vegetarian','Vegan')
    protein='tofu' if vegetarian else 'grilled chicken'
    meals=[{'Meal':'Breakfast','Draft idea':'Oatmeal with berries; portion to the agreed carbohydrate plan.'},
           {'Meal':'Lunch','Draft idea':f'{protein.title()} with vegetables and a portion of brown rice.'},
           {'Meal':'Dinner','Draft idea':f'Vegetable soup with {protein} and a portion of whole-grain bread; choose lower-sodium ingredients.'}]
    return {'status':'Meal ideas for clinician/dietitian review','advice':advice,'meals':meals,'sources':[DASH_SOURCE,'https://www.cdc.gov/diabetes/healthy-eating/diabetes-meal-planning.html']}

def discharge_text(encounter):
    d=encounter['data']; closed=encounter['stage']=='Discharged'
    lines=['# '+('Discharge summary' if closed else 'DRAFT — discharge summary'),
           f"Patient: {encounter['name']} | Reference: {encounter['mrn']}",f"Encounter: {encounter['id']}",
           f"Admission: {encounter['admitted_at']} | Discharge: {encounter.get('discharged_at') or 'Not finalized'}",
           '\n## Diagnosis',d.get('diagnosis') or 'Not recorded','\n## Hospital course',d.get('course') or 'Not recorded',
           '\n## Previous conditions',', '.join(d.get('conditions',[])) or 'None recorded (check history review)',
           '\n## Allergies',d.get('allergy_status','Unknown')+': '+d.get('allergies',''),
           '\n## Reconciled medications']
    for med in d.get('medications',[]):
        lines.append(f"- {med['name']}: {med.get('decision','Unreviewed')}. {med.get('discharge_instruction') or 'No discharge instructions recorded.'}")
    lines += ['\n## Nutrition instructions',d.get('nutrition_note') or 'Not recorded','\n## Follow-up',d.get('followup') or 'Not recorded',
              '\n## Return precautions',d.get('return_precautions') or 'Not recorded','\n## Reviewing clinician',d.get('clinician') or 'Not recorded',
              '\nThis prototype records clinician-entered instructions. AI drafts are not prescriptions or independent discharge authorization.']
    from india_documentation import summary as india_summary
    lines += india_summary(encounter)
    return '\n\n'.join(lines)
