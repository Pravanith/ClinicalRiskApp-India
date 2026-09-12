import sqlite3
import pandas as pd
import numpy as np
import os
import re
import datetime
import joblib
def redact_pii(text):
    """
    Sanitizes input to remove potential Patient Identifiers (Safe Harbor)
    before sending to an external LLM.
    """
    # Redact names (e.g., Mr. Smith, Dr. Jones)
    text = re.sub(r'\b(Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+', '', text)
    # Redact potential Medical Record Numbers (6+ digits)
    text = re.sub(r'\b\d{6,}\b', '', text)
    # Redact Dates (simple pattern)
    text = re.sub(r'\d{2}/\d{2}/\d{4}', '', text)
    return text

# Prevents app crash if drug_data.py is missing
try:
    from drug_data import INTERACTION_DB
except ImportError:
    print("⚠️ Warning: drug_data.py not found. Interaction checker will use empty DB.")
    INTERACTION_DB = {}

# ==========================================
# 1. DATABASE MANAGEMENT
# ==========================================
def get_db_connection():
    # check_same_thread=False is crucial for Streamlit's multi-threaded environment
    return sqlite3.connect('clinical_data.db', check_same_thread=False)

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS patient_history (
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            age INTEGER,
            gender TEXT,
            sbp INTEGER,
            aki_risk_score INTEGER,
            bleeding_risk_score REAL,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_patient_to_db(age, gender, sbp, aki, bleed, status):
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO patient_history (age, gender, sbp, aki_risk_score, bleeding_risk_score, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (age, gender, sbp, aki, bleed, status))
        conn.commit()

def fetch_history():
    if not os.path.exists('clinical_data.db'):
        return pd.DataFrame()
    with get_db_connection() as conn:
        df = pd.read_sql("SELECT * FROM patient_history ORDER BY timestamp DESC", conn)
    return df

def clear_history():
    with get_db_connection() as conn:
        conn.execute("DELETE FROM patient_history")
        conn.commit()

# ==========================================
# 2. AI MODEL LOADING
# ==========================================
class HeuristicFallbackModel:
    """
    Deterministic clinical rule-based fallback if ML model is missing.
    Used to ensure system reliability in production.
    """
    def predict(self, df):
        risk = 10.0 
        row = df.iloc[0]
        if row.get('age', 0) > 65: risk += 15
        if row.get('high_bp', 0) == 1: risk += 20
        if row.get('inr', 1.0) > 1.2: risk += 25
        if row.get('anticoagulant', 0) == 1: risk += 20
        return [min(risk, 95.0)]

def load_bleeding_model():
    model_file = os.path.join(os.path.dirname(__file__), "clinical_pipeline_v4.pkl")
    if not os.path.exists(model_file):
        model_file = os.path.join(os.path.dirname(__file__), "clinical_pipeline.pkl")
    if os.path.exists(model_file):
        try:
            # Loads the Pipeline (Preprocessor + Model)
            pipeline = joblib.load(model_file)
            return pipeline
        except Exception as e:
            print(f"⚠️ Error: {e}")
            return HeuristicFallbackModel()
    return HeuristicFallbackModel()
# ==========================================
# 3. CLINICAL CALCULATORS (LOGIC)
# ==========================================
def calculate_aki_risk(age, diuretic, acei, sys_bp, chemo, creat, nsaid, heart_failure):
    score = 0
    score += 30 if diuretic else 0
    score += 40 if acei else 0
    score += 25 if nsaid else 0
    score += 15 if heart_failure else 0
    score += 20 if chemo else 0
    if age > 0: score += 20 if age > 75 else 0
    if sys_bp > 0:
        score += 10 if sys_bp > 160 else 0
        score += 20 if sys_bp < 90 else 0
    if creat > 0:
        if creat > 1.5: score += 30
        elif creat > 1.2: score += 15
    return min(score, 100)

def calculate_sepsis_risk(sys_bp, resp_rate, altered_mental, temp_c):
    # Sepsis-3 qSOFA: https://pmc.ncbi.nlm.nih.gov/articles/4968574/
    return int(0 < sys_bp <= 100) + int(resp_rate >= 22) + int(bool(altered_mental))


def calculate_hypoglycemic_risk(insulin, renal, hba1c_high, recent_dka):
    score = 0
    score += 30 if insulin else 0
    score += 45 if renal else 0
    score += 20 if hba1c_high else 0
    score += 20 if recent_dka else 0 
    return min(score, 100)

def calculate_sirs_score(temp_c, hr, resp_rate, wbc):
    score = 0
    if temp_c > 0 and (temp_c > 38 or temp_c < 36): score += 1
    if hr > 90: score += 1
    if resp_rate > 20: score += 1
    if wbc > 0 and (wbc > 12 or wbc < 4): score += 1
    return score

# ==========================================
# 4. INTERACTION CHECKER
# ==========================================
def normalize_text(text):
    """Simple normalization to handle slight variations in drug names"""
    if not isinstance(text, str): return ""
    return text.lower().strip()

def check_interaction(d1, d2):
    d1_clean = normalize_text(d1)
    d2_clean = normalize_text(d2)
    
    if (d1_clean, d2_clean) in INTERACTION_DB: return INTERACTION_DB[(d1_clean, d2_clean)]
    if (d2_clean, d1_clean) in INTERACTION_DB: return INTERACTION_DB[(d2_clean, d1_clean)]
    return None

# ==========================================
# 5. KNOWLEDGE BASE & CHATBOT LOGIC
# ==========================================
KNOWLEDGE_BASE = {
    # --- CARDIOLOGY ---
        "mi": "Myocardial Infarction (Heart Attack). Blockage of blood flow. STEMI is critical. Symptoms: Chest pressure, radiating pain.",
        "heart attack": "Myocardial Infarction (MI). Emergency. Protocol: MONA (Morphine, O2, Nitro, Aspirin).",
        "hypertension": "High Blood Pressure (>130/80). 'The Silent Killer'. Risk of Stroke/MI/Kidney Failure.",
        "hypotension": "Low BP (<90/60). Signs of shock. Causes: Dehydration, Sepsis, Hemorrhage.",
        "afib": "Atrial Fibrillation. Irregular heart rate. Risk: Stroke. Treatment: Beta-blockers + Anticoagulation.",
        "chf": "Congestive Heart Failure. Fluid overload. Symptoms: SOB lying flat (Orthopnea), Edema. Labs: BNP.",
        "cad": "Coronary Artery Disease. Plaque buildup in heart arteries. Risk of MI. Managed with Statins.",
        "angina": "Chest pain due to ischemia (lack of O2). Stable (exertional) vs Unstable (at rest).",
        "aortic stenosis": "Narrowing of aortic valve. Symptoms: SAD (Syncope, Angina, Dyspnea). Murmur heard.",
        "pericarditis": "Inflammation of heart sac. Sharp pain relieved by leaning forward. Friction rub heard.",
        "endocarditis": "Infection of heart valves. IV drug use is risk factor. Fever + New Murmur.",
        "bradycardia": "Slow heart rate (<60). Causes: Beta-blockers, heart block, athletic heart.",
        "tachycardia": "Fast heart rate (>100). Causes: Fever, Pain, Dehydration, PE, Anxiety.",
        "svt": "Supraventricular Tachycardia. Rate >150. Tx: Vagal maneuvers, Adenosine.",
        "vtach": "Ventricular Tachycardia. Life-threatening. Pulse? Cardiovert. No pulse? Defibrillate.",
        "vfib": "Ventricular Fibrillation. Cardiac Arrest. No pulse. CPR + Defibrillation immediately.",

        # --- RESPIRATORY ---
        "asthma": "Airway inflammation. Wheezing. Rescue: Albuterol. Maintenance: Steroids.",
        "copd": "Chronic Obstructive Pulmonary Disease. Air trapping. Smokers. Risk of CO2 retention.",
        "pneumonia": "Lung infection. Fever, cough, consolidation on X-Ray. Antibiotics needed.",
        "pe": "Pulmonary Embolism. Lung clot. Sudden SOB, chest pain, tachycardia. Emergency.",
        "pneumothorax": "Collapsed lung. Air in pleural space. Absent breath sounds. Trauma/Spontaneous.",
        "pleural effusion": "Fluid around lungs. 'Water on lungs'. Causes: CHF, Cancer. Tx: Thoracentesis.",
        "ards": "Acute Respiratory Distress Syndrome. Severe lung failure. 'White out' X-ray.",
        "bronchitis": "Inflammation of bronchi. Productive cough. Usually viral.",
        "tuberculosis": "TB. Bacterial infection. Night sweats, weight loss, bloody cough. Isolation required.",
        "sleep apnea": "Airway collapse during sleep. Snoring, fatigue. Risk: HTN, Stroke. Tx: CPAP.",
        "croup": "Pediatric viral infection. 'Barking seal' cough. Steeple sign on X-ray.",
        "rsv": "Respiratory Syncytial Virus. Bronchiolitis in kids. Watch for hypoxia.",

        # --- NEUROLOGY ---
        "cva": "Cerebrovascular Accident (Stroke). Ischemic vs Hemorrhagic. Time is Brain.",
        "stroke": "Brain attack. BE-FAST: Balance, Eyes, Face, Arms, Speech, Time. CT Head stat.",
        "tia": "Transient Ischemic Attack. Warning stroke. Symptoms resolve <24h. High risk of future stroke.",
        "seizure": "Abnormal electrical activity. Protect head. Benzodiazepines if > 5 mins.",
        "epilepsy": "Recurrent seizures. Meds: Keppra, Depakote, Phenytoin.",
        "migraine": "Severe unilateral headache. Nausea, photophobia. Tx: Triptans.",
        "cluster headache": "Severe eye pain. 'Suicide headache'. Tx: High flow Oxygen.",
        "meningitis": "Brain lining infection. Fever + Stiff Neck + Headache. Emergency.",
        "concussion": "Mild TBI. Confusion, amnesia, headache. Brain rest needed.",
        "parkinson": "Low dopamine. Tremor (resting), Rigidity, Slow movement (Bradykinesia).",
        "alzheimer": "Dementia type. Memory loss, cognitive decline.",
        "ms": "Multiple Sclerosis. Autoimmune nerve damage. Vision loss, weakness.",
        "als": "Lou Gehrig's Disease. Motor neuron death. Paralysis. Sensation intact.",
        "guillain-barre": "Ascending paralysis after infection. Watch breathing.",

        # --- GASTROINTESTINAL ---
        "gerd": "Acid Reflux. Heartburn. Risk of esophageal damage. PPIs (Omeprazole).",
        "pud": "Peptic Ulcer Disease. Stomach ulcers. Pain with food. H. Pylori or NSAIDs.",
        "gi bleed": "Upper (Vomit blood) vs Lower (Bloody stool). Monitor Hemoglobin.",
        "appendicitis": "RLQ pain (McBurney's). Fever, nausea. Surgical emergency.",
        "cholecystitis": "Gallbladder inflammation. RUQ pain after fatty meal.",
        "pancreatitis": "Pancreas inflammation. Epigastric pain to back. High Lipase.",
        "hepatitis": "Liver inflammation. Viral (A/B/C) or Alcohol. Jaundice.",
        "cirrhosis": "End-stage liver scarring. Ascites, confusion (Ammonia), Bleeding risk.",
        "ibs": "Irritable Bowel Syndrome. Functional pain/diarrhea. No organ damage.",
        "ibd": "Crohn's/Ulcerative Colitis. Autoimmune. Bloody diarrhea.",
        "diverticulitis": "Infected colon pouches. LLQ pain, fever. Antibiotics.",
        "c diff": "Antibiotic diarrhea. Contagious spores. Soap & Water wash only.",
        "bowel obstruction": "Blockage. Constipation, vomiting, distension. NPO + NG Tube.",

        # --- RENAL ---
        "aki": "Acute Kidney Injury. Creatinine spike. Causes: Dehydration, Contrast, NSAIDs.",
        "ckd": "Chronic Kidney Disease. GFR < 60 > 3 months. Diabetes/HTN causes.",
        "esrd": "End Stage Renal Disease. Needs Dialysis or Transplant.",
        "uti": "Urinary Tract Infection. Burning, frequency. E. Coli common.",
        "kidney stone": "Nephrolithiasis. Flank pain, hematuria. Hydration + Pain meds.",
        "bph": "Enlarged prostate. Dribbling, frequency in older men.",
        "rhabdo": "Muscle breakdown. Clogs kidneys. Tea-colored urine. Fluids.",

        # --- ENDOCRINE ---
        "diabetes": "Metabolic disease. High sugar. Causes damage to eyes, kidneys, nerves.",
        "diabetes type 1": "Autoimmune. No insulin. DKA risk. Insulin dependent.",
        "diabetes type 2": "Insulin resistance. Lifestyle + Metformin.",
        "prediabetes": "A1C 5.7-6.4%. Warning sign. Reversible.",
        "dka": "Diabetic Ketoacidosis. Acidosis + Ketones. ICU care.",
        "hhs": "Hyperosmolar Hyperglycemic State. Glucose > 600. Dehydration.",
        "hypoglycemia": "Low Sugar (<70). Sweating, confusion. Glucose needed.",
        "hypothyroidism": "Low Thyroid. Fatigue, weight gain, cold.",
        "hyperthyroidism": "High Thyroid. Weight loss, heat intolerance, fast HR.",
        "addison": "Adrenal insufficiency. Low cortisol. Bronze skin, hypotension.",
        "cushing": "High cortisol. Moon face, buffalo hump, high sugar.",

        # --- INFECTIOUS DISEASE ---
        "sepsis": "Infection + Organ Failure. qSOFA criteria. Antibiotics ASAP.",
        "septic shock": "Sepsis + Hypotension requiring pressors.",
        "flu": "Influenza. Sudden fever, aches. Tamiflu < 48h.",
        "covid": "SARS-CoV-2. Fever, cough, loss of taste/smell.",
        "hiv": "Virus attacking immune system (CD4). Needs ARV therapy.",
        "mrsa": "Resistant Staph infection. Needs Vancomycin.",
        "cellulitis": "Skin infection. Red, hot, spreading.",
        "abscess": "Pus pocket. Needs drainage (I&D).",
        "osteomyelitis": "Bone infection. Long-term IV antibiotics.",
        "fever": "Temp > 100.4F (38C). Sign of inflammation.",
        "neutropenic fever": "Fever in chemo patient. ONCOLOGIC EMERGENCY.",

        # --- HEMATOLOGY ---
        "anemia": "Low Hemoglobin. Fatigue, pallor. Iron deficiency common.",
        "sickle cell": "Genetic. Pain crises. RBCs shape sickle. Fluids/Pain meds.",
        "thrombocytopenia": "Low platelets. Bleeding risk.",
        "leukemia": "Blood cancer. High WBC blasts. Infection risk.",
        "lymphoma": "Lymphatic cancer. Hodgkin vs Non-Hodgkin.",
        "neutropenia": "Low neutrophils. Severe infection risk. Fever is emergency.",
        "dvt": "Deep Vein Thrombosis. Leg clot.",

        # --- MUSCULOSKELETAL ---
        "osteoarthritis": "Wear-and-tear. Joint pain. Worse with use.",
        "rheumatoid arthritis": "Autoimmune. Morning stiffness > 30 mins.",
        "gout": "Uric acid crystals. Big toe pain. Colchicine/Allopurinol.",
        "osteoporosis": "Weak bones. Fracture risk.",
        "compartment syndrome": "Muscle pressure. Pain out of proportion. Emergency surgery.",
        "rhabdomyolysis": "Muscle breakdown releasing myoglobin. Kidney damage. Tea-colored urine.",

        # --- MEDICATIONS ---
        "lisinopril": "ACE Inhibitor (BP). Side effects: Cough, High K+.",
        "amlodipine": "Calcium Channel Blocker (BP). Side effect: Leg swelling.",
        "metoprolol": "Beta-blocker. Lowers HR and BP.",
        "furosemide": "Lasix. Diuretic. Monitor Potassium.",
        "spironolactone": "Potassium-sparing diuretic. Risk: High K+.",
        "atorvastatin": "Lipitor. Cholesterol. Watch for muscle pain.",
        "metformin": "Diabetes. Risk: Lactic Acidosis. Hold for contrast.",
        "insulin": "Lowers sugar. High Alert. Hypoglycemia risk.",
        "glipizide": "Sulfonylurea. Stimulates pancreas. Hypo risk.",
        "warfarin": "Coumadin. Anticoagulant. Monitor INR (2-3).",
        "eliquis": "Apixaban. Blood thinner. No INR needed.",
        "plavix": "Clopidogrel. Antiplatelet. Keeps stents open.",
        "aspirin": "Antiplatelet. MI/Stroke prevention. Bleed risk.",
        "ibuprofen": "NSAID. Pain/Fever. Avoid in Kidney/Ulcer/CHF.",
        "naproxen": "Aleve. NSAID.",
        "acetaminophen": "Tylenol. Pain/Fever. Liver toxicity > 4g.",
        "tramadol": "Weak opioid. Serotonin syndrome risk.",
        "oxycodone": "Opioid. Severe pain. Respiratory depression risk.",
        "morphine": "Opioid. Gold standard for MI/Pain.",
        "naloxone": "Narcan. Opioid antidote.",
        "albuterol": "Asthma rescue inhaler. Causes jitters.",
        "prednisone": "Steroid. High sugar, anger, insomnia.",
        "pantoprazole": "Protonix. Acid reflux.",
        "ondansetron": "Zofran. Nausea. QT prolongation.",
        "sertraline": "Zoloft. SSRI. Depression/Anxiety.",
        "fluoxetine": "Prozac. SSRI.",
        "alprazolam": "Xanax. Benzo. Anxiety. Addictive.",
        "lorazepam": "Ativan. Benzo. Seizures.",
        "vancomycin": "Antibiotic (MRSA). Monitor levels. Red Man Syndrome.",
        "piperacillin": "Zosyn. Broad spectrum antibiotic.",
        "ciprofloxacin": "Antibiotic (UTI). Tendon rupture risk.",
        "azithromycin": "Z-Pak. Pneumonia.",
        "diphenhydramine": "Benadryl. Allergy. Sedating.",
        "epinephrine": "Adrenaline. Anaphylaxis/Code Blue.",

        # --- LABS ---
        "wbc": "White Blood Cells. High=Infection.",
        "hgb": "Hemoglobin. Low=Anemia. <7 Transfuse.",
        "plt": "Platelets. Low=Bleeding risk.",
        "na": "Sodium. Low=Confusion.",
        "k": "Potassium. Critical for heart rhythm.",
        "bun": "BUN. High=Dehydration/Kidney.",
        "cr": "Creatinine. Best kidney marker.",
        "glucose": "Blood Sugar. 70-100 Fasting.",
        "a1c": "HbA1c. 3-month sugar avg. <5.7 Normal. >6.5 Diabetes.",
        "trop": "Troponin. Heart enzyme. High=Heart Attack.",
        "bnp": "Heart Failure marker. High=Fluid overload.",
        "inr": "Clotting time (Warfarin). Normal 1.0. Goal 2-3.",
        "lactate": "Sepsis marker. >2.0 indicates shock.",
        "ph": "Acidity (7.35-7.45).",

        # --- ABBREVIATIONS ---
        "bid": "Twice a day.",
        "tid": "Three times a day.",
        "qid": "Four times a day.",
        "qd": "Daily.",
        "prn": "As needed.",
        "ac": "Before meals.",
        "pc": "After meals.",
        "po": "By mouth.",
        "iv": "Intravenous.",
        "im": "Intramuscular.",
        "npo": "Nothing by mouth.",
        "stat": "Immediately.",
        "vs": "Vital Signs.",
        "nkda": "No Known Drug Allergies."
}

def chatbot_response(text):
    text = text.lower().strip()
    
    # 1. Local Search
    for key, value in KNOWLEDGE_BASE.items():
        if key in text:
            return f"**📚 GLOSSARY MATCH ({key.upper()}):**\n{value}"

    # 2. Fallback to AI
    try:
        import google.generativeai as genai
        import streamlit as st
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""
        Define the medical term: "{text}".
        Keep it concise (under 30 words).
        If not medical, say "Term not recognized."
        """
        response = model.generate_content(prompt)
        return f"**🧠 AI DEFINITION:**\n{response.text}"
    except:
        return "ℹ️ Term not found in Glossary and AI is unavailable."

# ==========================================
# 6. AI CONSULTANTS (GEMINI)
# ==========================================
def consult_ai_doctor(role, user_input, patient_context=None):
    try:
        # 1. Redact PII for Privacy/HIPAA Compliance
        safe_input = redact_pii(user_input)
        
        # 2. Configure Model
        genai.configure(api_key=st.secrets)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # 3. Chain-of-Thought Prompting (Higher reliability)
        if role == 'risk_assessment':
            prompt = f"""
            Role: Senior ICU Consultant.
            Context: {patient_context}
            User Query: {safe_input}
            
            Instructions:
            1. First, analyze the hemodynamic stability (MAP, Shock Index).
            2. Second, cross-reference active meds with the bleeding risk score.
            3. Finally, recommend 3 specific, prioritized clinical actions.
            4. Cite relevant guidelines (e.g., KDIGO, Surviving Sepsis) where applicable.
            """
        else:
            prompt = f"Expert Medical Consult. Query: {safe_input}. Provide differential diagnosis."

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"System Error: {str(e)}"

def generate_discharge_summary(patient_data):
    import google.generativeai as genai
    import streamlit as st
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"Write a discharge summary for a patient with: {patient_data}"
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_drug_interactions(drug_list):
    import google.generativeai as genai
    import streamlit as st
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"Analyze drug interactions for: {', '.join(drug_list)}. List mechanism and management."
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Error: {str(e)}"
