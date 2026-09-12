import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import backend as bk
import random
import datetime
import json
from pathlib import Path
from voice_ui import render_voice_entry
from care_ui import render_hospital
import admission

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Clinical Risk Monitor", 
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS (Called immediately after page config)
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #212529;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏥 Clinical Risk Monitor")
st.caption("Clinical workspace • Voice-assisted intake • Risk assessment • Medication review")
st.info("Educational prototype. Predictions use synthetic data; review all observations and outputs before interpretation.")
st.divider()

# Helper for file timestamps
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")

# Initialize Database
st.warning("PUBLIC DEMO — use fictional patients only. Records in this demo may be visible to other visitors and may be reset. Do not enter real patient information.")
bk.init_db()

# Load AI Model
try:
    bleeding_model = bk.load_bleeding_model()
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

# Session State Initialization
if 'patient_data' not in st.session_state:
    st.session_state['patient_data'] = {}
if 'entered_app' not in st.session_state:
    st.session_state['entered_app'] = False

# ---------------------------------------------------------
# 2. UI MODULES
# ---------------------------------------------------------

# --- COVER PAGE ---
def render_cover_page():
    st.markdown("<h1 style='text-align: center;'>🛡️ Clinical Risk Monitor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI-Driven Pharmacovigilance System</p>", unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns([1, 2, 1])
    if c2.button("🚀 Launch Dashboard", use_container_width=True, type="primary"):
        st.session_state['entered_app'] = True
        st.rerun()

def invalidate_analysis():
    st.session_state.pop('analysis_results', None)
    st.session_state['patient_data'] = {}

# --- MODULE 1: RISK CALCULATOR ---
def render_risk_calculator():
    st.subheader("Acute Risk Calculator")
    if isinstance(bleeding_model, bk.HeuristicFallbackModel):
        st.warning("ML model unavailable: bleeding output uses an unvalidated heuristic fallback.")
    else:
        st.caption("Bleeding model loaded • Synthetic-data demonstration")
    report_path = Path(__file__).parent / 'model_report.json'
    if report_path.exists() and not isinstance(bleeding_model, bk.HeuristicFallbackModel):
        with st.expander("Model performance and limitations"):
            try:
                report = json.loads(report_path.read_text())
                scores = report['calibrated']
                cols = st.columns(3)
                cols[0].metric('Holdout ROC-AUC', f"{scores['roc_auc']:.3f}")
                cols[1].metric('Brier score', f"{scores['brier_score']:.3f}")
                cols[2].metric('Holdout observations', report['holdout_rows'])
                st.caption('Gradient boosting with sigmoid calibration. These are synthetic-data holdout results, not clinical validation. Lower Brier scores indicate lower probability prediction error.')
                st.write('Model inputs: ' + ', '.join(report['features']))
            except (ValueError, KeyError, TypeError):
                st.caption('Model evaluation report is unavailable.')
    render_voice_entry()
    st.caption("Enter current observations. Zero represents missing data; complete required fields before analysis.")
    
    # --- INPUTS CONTAINER ---
    with st.container(border=True):
        st.markdown("#### 📝 Patient Data Entry")
        
        with st.container():
            
            # Split Screen Layout
            col_left, col_right = st.columns([1, 1], gap="medium")
            
            # --- LEFT COLUMN: Demographics & Vitals ---
            with col_left:
                st.markdown("##### 👤 Patient Profile")
                l1, l2 = st.columns(2)
                age = l1.number_input("Age (Years)", min_value=0, max_value=120, value=0, key="risk_age", on_change=invalidate_analysis)
                gender = l2.selectbox("Gender", ["Male", "Female"], key="risk_gender", on_change=invalidate_analysis)
                
                w_val, w_unit = st.columns([2, 1]) 
                weight_input = w_val.number_input("Weight", 0.0, 400.0, 0.0, key="risk_weight_input", on_change=invalidate_analysis)
                weight_scale = w_unit.selectbox("Unit", ["kg", "lbs"], key="risk_w_unit", on_change=invalidate_analysis)
                height = st.number_input("Height (cm)", 0, 250, 0, key="risk_height", on_change=invalidate_analysis)
                
                # Weight Logic
                weight_kg = weight_input * 0.453592 if weight_scale == "lbs" else weight_input
                if height > 0:
                    bmi = weight_kg / ((height/100)**2)
                else:
                    bmi = 0.0

                st.markdown("##### 🩺 Vitals")
                v1, v2 = st.columns(2)
                sys_bp = v1.number_input("Systolic BP (Normal: 110-120)", 0, 300, 0, key="risk_sys_bp", on_change=invalidate_analysis)
                dia_bp = v2.number_input("Diastolic BP (Normal: 70-80)", 0, 200, 0, key="risk_dia_bp", on_change=invalidate_analysis)
                
                v3, v4 = st.columns(2)
                hr = v3.number_input("Heart Rate (Normal: 60-100)", 0, 300, 0, key="risk_hr", on_change=invalidate_analysis)
                resp_rate = v4.number_input("Resp Rate (Normal: 12-20)", 0, 60, 0, key="risk_resp_rate", on_change=invalidate_analysis)
                
                v5, v6 = st.columns(2)
                def temperature_changed():
                    f = st.session_state.get('risk_temp_f')
                    st.session_state['risk_temp_c'] = (f-32)*5/9 if f is not None else 0.0
                    invalidate_analysis()
                if 'risk_temp_f' not in st.session_state:
                    c = st.session_state.get('risk_temp_c', 0)
                    st.session_state['risk_temp_f'] = c*9/5+32 if c else None
                v5.number_input("Temperature °F", min_value=32.0, max_value=113.0, value=None, step=0.1, key="risk_temp_f", on_change=temperature_changed, placeholder="Not recorded")
                temp_c = st.session_state.get('risk_temp_c', 0.0)
                o2_sat = v6.number_input("O2 Sat % (Normal: >95%)", 0, 100, 0, key="risk_o2_sat", on_change=invalidate_analysis)

            # --- RIGHT COLUMN: Labs & History ---
            with col_right:
                st.markdown("##### 🧪 Critical Labs")
                
                lab1, lab2 = st.columns(2)
                creat = lab1.number_input("Creatinine (0.6-1.2 mg/dL)", 0.0, 20.0, 0.0, key="risk_creat", on_change=invalidate_analysis)
                bun = lab2.number_input("Blood Urea Nitrogen (7-20)", 0, 100, 0, key="risk_bun", on_change=invalidate_analysis)
                
                lab3, lab4 = st.columns(2)
                potassium = lab3.number_input("Potassium (3.5-5.0 mmol/L)", 0.0, 10.0, 0.0, key="risk_potassium", on_change=invalidate_analysis)
                glucose = lab4.number_input("Glucose (70-100 mg/dL)", 0, 1000, 0, key="risk_glucose", on_change=invalidate_analysis)
                
                lab5, lab6 = st.columns(2)
                wbc = lab5.number_input("WBC (4.5-11.0 10^9/L)", 0.0, 50.0, 0.0, key="risk_wbc", on_change=invalidate_analysis)
                hgb = lab6.number_input("Hemoglobin (13.5-17.5 g/dL)", 0.0, 20.0, 0.0, key="risk_hgb", on_change=invalidate_analysis)
                
                lab7, lab8 = st.columns(2)
                platelets = lab7.number_input("Platelets (150-450 10^9/L)", 0, 1000, 0, key="risk_platelets", on_change=invalidate_analysis)
                inr = lab8.number_input("INR (Clotting Time) [0.9-1.1]", 0.0, 10.0, 0.0, key="risk_inr", on_change=invalidate_analysis)
                
                lactate = st.number_input("Lactate (Normal: < 2.0 mmol/L)", 0.0, 20.0, 0.0, key="risk_lactate", on_change=invalidate_analysis)

                st.markdown("##### 📋 Medical History")
                h1, h2 = st.columns(2)
                anticoag = h1.checkbox("Anticoagulant Use", key="risk_anticoag", on_change=invalidate_analysis)
                liver_disease = h2.checkbox("Liver Disease", key="risk_liver_disease", on_change=invalidate_analysis)
                
                h3, h4 = st.columns(2)
                heart_failure = h3.checkbox("Heart Failure", key="risk_heart_failure", on_change=invalidate_analysis)
                gi_bleed = h4.checkbox("History of GI Bleed", key="risk_gi_bleed", on_change=invalidate_analysis)
                
                m1, m2 = st.columns(2)
                nsaid = m1.checkbox("NSAID Use", key="risk_nsaid", on_change=invalidate_analysis)
                active_chemo = m2.checkbox("Active Chemo", key="risk_active_chemo", on_change=invalidate_analysis)
                
                m3, m4 = st.columns(2)
                diuretic = m3.checkbox("Diuretic Use", key="risk_diuretic", on_change=invalidate_analysis)
                acei = m4.checkbox("ACEi/ARB", key="risk_acei", on_change=invalidate_analysis)
                
                m5, m6 = st.columns(2)
                insulin = m5.checkbox("Insulin", key="risk_insulin", on_change=invalidate_analysis)
                hba1c_high = m6.checkbox("Uncontrolled Diabetes", key="risk_hba1c_high", on_change=invalidate_analysis)
                
                altered_mental = st.checkbox("Altered Mental Status (Confusion)", key="risk_altered_mental", on_change=invalidate_analysis)
                pain = 0

            additional, admission_checks = admission.render_inputs('risk_', on_change=invalidate_analysis)
            st.write("") 
            submitted = st.button("🚀 Run Clinical Analysis", type="primary", use_container_width=True)

    # --- LOGIC & RESULTS ---
    if submitted:
        required = {"Age": age, "Systolic BP": sys_bp, "Diastolic BP": dia_bp,
                    "Heart rate": hr, "Respiratory rate": resp_rate, "Temperature": temp_c,
                    "Creatinine": creat, "INR": inr, "WBC": wbc, "Glucose": glucose}
        missing = [name for name, value in required.items() if value <= 0]
        if missing or dia_bp >= sys_bp:
            st.session_state.pop('analysis_results', None)
            st.session_state['patient_data'] = {}
            st.error('Full legacy analysis needs: ' + ', '.join(missing) + '. These are calculator inputs, not a recommendation to order every test.' if missing else 'Diastolic BP must be below systolic BP.')
            admission.render_summary({**additional, 'o2_sat':o2_sat, 'weight':weight_kg}, admission_checks)
            return
        final_temp_c = temp_c 
        
        # Calculate Hemodynamics
        if sys_bp > 0:
            map_val = (sys_bp + (2 * dia_bp)) / 3 
            pulse_pressure = sys_bp - dia_bp
            shock_index = hr / sys_bp if sys_bp > 0 else 0
        else:
            map_val = 0
            pulse_pressure = 0
            shock_index = 0
            
        bun_creat_ratio = bun / creat if creat > 0 else 0
        is_high_bp = 1 if sys_bp > 140 else 0
        
# --- GLOBAL ZERO CHECK ---
        if age > 0 and sys_bp > 0:
            # ---------------------------------------------------------
            # NEW PIPELINE LOGIC
            # ---------------------------------------------------------
            # We explicitly map form variables to the EXACT feature names 
            # defined in your 'train_model_pro.py' script.
            input_df = pd.DataFrame({
                'age': [age],
                'inr': [inr],
                'systolic_bp': [sys_bp],         # Renamed to match training feature 'systolic_bp'
                'anticoagulant': [1 if anticoag else 0],
                'gender': [gender],              # Pass raw "Male" or "Female" (Pipeline handles encoding)
                'liver_disease': [1 if liver_disease else 0]
            })

            # Get Probability of Critical Risk (Class 1)
            # The pipeline handles scaling & OneHotEncoding automatically.
            # predict_proba returns [[Prob_Class_0, Prob_Class_1]]
            try:
                pred_bleeding = bleeding_model.predict_proba(input_df)[0][1] * 100
            except AttributeError:
                # Fallback if model is still the old Regressor (during transition)
                pred_bleeding = float(bleeding_model.predict(input_df)[0])

            # ---------------------------------------------------------
            # EXISTING LOGIC (Unchanged)
            # ---------------------------------------------------------
            pred_aki = bk.calculate_aki_risk(age, diuretic, acei, sys_bp, active_chemo, creat, nsaid, heart_failure)
            pred_sepsis = bk.calculate_sepsis_risk(sys_bp, resp_rate, altered_mental, final_temp_c)
            pred_hypo = bk.calculate_hypoglycemic_risk(insulin, (creat>1.3), hba1c_high, False)
            sirs_score = bk.calculate_sirs_score(final_temp_c, hr, resp_rate, wbc)
            
            # HAS-BLED Calculator (Heuristic Check)
            has_bled = 0
            if sys_bp > 160: has_bled += 1
            if creat > 2.2 or liver_disease: has_bled += 1
            if gi_bleed: has_bled += 1
            if inr > 1.0: has_bled += 1
            if age > 65: has_bled += 1
            if nsaid or anticoag: has_bled += 1

        else:
            pred_bleeding = 0.0
            pred_aki = 0
            pred_sepsis = 0
            pred_hypo = 0
            sirs_score = 0
            has_bled = 0
            
        # Determine Status
        status_calc = 'Critical' if (pred_bleeding > 50 or pred_aki > 50 or pred_sepsis >= 2) else 'Stable'
        
        # Save to Database
        bk.save_patient_to_db(age, gender, sys_bp, int(pred_aki), float(pred_bleeding), status_calc)
        
        # Save ALL relevant data to session state for safe retrieval
        st.session_state['patient_data'] = {
            **additional, 'admission_checks': admission_checks,
            'id': f"Patient-{age}-{int(sys_bp)}", 
            'age': age, 'gender': gender, 'weight': weight_kg,
            'sys_bp': sys_bp, 'dia_bp': dia_bp, 'hr': hr, 'resp_rate': resp_rate, 
            'temp_c': temp_c, 'o2_sat': o2_sat, 'pain': pain,
            'creat': creat, 'potassium': potassium, 'inr': inr, 'bun': bun,
            'wbc': wbc, 'hgb': hgb, 'platelets': platelets, 'lactate': lactate, 'glucose': glucose,
            'bleeding_risk': float(pred_bleeding), 'aki_risk': int(pred_aki),
            'sepsis_risk': int(pred_sepsis), 'hypo_risk': int(pred_hypo),
            'sirs_score': sirs_score, 'status': status_calc, 'map_val': map_val, 'bmi': bmi, 'has_bled': has_bled,
            'shock_index': shock_index, 'pulse_pressure': pulse_pressure, 'bun_creat_ratio': bun_creat_ratio,
            # SAVE BOOLEANS FOR EXPLANATION LOGIC
            'anticoag': anticoag, 'liver_disease': liver_disease, 'diuretic': diuretic,
            'acei': acei, 'heart_failure': heart_failure, 'hba1c_high': hba1c_high,
            'altered_mental': altered_mental
        }
        
        st.session_state['analysis_results'] = st.session_state['patient_data']

    # Streamlit discards widget values when navigating away; never show an old
    # assessment beside a newly empty form after returning to this module.
    if not submitted and 'analysis_results' in st.session_state:
        previous = st.session_state['analysis_results']
        current_inputs = {'age': age, 'sys_bp': sys_bp, 'dia_bp': dia_bp, 'hr': hr,
                          'resp_rate': resp_rate, 'temp_c': temp_c, 'creat': creat,
                          'inr': inr, 'wbc': wbc, 'glucose': glucose, **additional}
        if any(previous.get(key) != value for key, value in current_inputs.items()):
            invalidate_analysis()

    # --- RESULTS DISPLAY ---
    if 'analysis_results' in st.session_state:
        res = st.session_state['analysis_results']
        
        st.divider()
        st.subheader("📊 Risk Stratification Results")
        
        # ROW 1: Major Risks
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("🩸 Bleeding model estimate", f"{res['bleeding_risk']:.1f}%",
                 "High model estimate" if res['bleeding_risk'] > 50 else "Model estimate", delta_color="off", help="Synthetic-data model estimate; not a validated clinical probability.")
        r2.metric("💧 AKI Risk", f"{res['aki_risk']}/100",
                 "High heuristic" if res['aki_risk'] > 50 else "Heuristic only", delta_color="off", help="Local heuristic score, not a KDIGO diagnosis or probability.")
        r3.metric("qSOFA", f"{res['sepsis_risk']}/3",
                 "Review required" if res['sepsis_risk'] >= 2 else "Does not rule out sepsis", delta_color="off", help="qSOFA is a clinical score, not a sepsis probability.")
        
        # Smart Glycemic Metric
        current_gluc = res.get('glucose', 0)
        if current_gluc > 180:
             r4.metric("🍬 Glycemia", f"{int(current_gluc)} mg/dL", "Hyper (High)", delta_color="inverse")
        elif current_gluc > 0 and current_gluc < 70:
             r4.metric("🍬 Glycemia", f"{int(current_gluc)} mg/dL", "Hypo (Low)", delta_color="inverse")
        else:
             r4.metric("Hypoglycemia heuristic", f"{res.get('hypo_risk', 0)}/100", "Not a probability", delta_color="off")

        # ROW 2: Advanced Hemodynamics
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("MAP", f"{int(res.get('map_val', 0))} mmHg", help="Mean Arterial Pressure")
        h2.metric("⚡ SIRS Score", f"{res.get('sirs_score', 0)}/4", help="Inflammatory Response")
        h3.metric("💔 Shock Index", f"{res.get('shock_index', 0):.2f}", "Review" if res.get('shock_index', 0) > 0.9 else "Derived value", delta_color="off", help="HR / SBP")
        h4.metric("💓 Pulse Pressure", f"{int(res.get('pulse_pressure', 0))}", "Wide" if res.get('pulse_pressure', 0) > 60 else "Derived value", delta_color="off", help="SBP - DBP")

        st.divider()

        admission.render_summary(res, res.get('admission_checks', {}))

  #   # [Image of sepsis pathophysiology diagram]

        # --- EXPLANATION SECTION (Now correctly indented) ---
        with st.expander("🧠 Clinical Logic: Why is this patient Critical?"):
            # Create 4 tabs to explain different risk categories
            tab1, tab2, tab3, tab4 = st.tabs(["Bleeding Risk", "AKI Risk", "Sepsis (qSOFA)", "Hemodynamics"])
        
            # --- TAB 1: BLEEDING RISK ---
            with tab1:
                st.write("### Factors driving the Bleeding Risk Score:")
                if res.get('inr', 0) > 3.5:
                    st.error(f"• **Critical INR ({res.get('inr')}):** +40 points (Major Driver)")
                if res.get('anticoag'): 
                    st.warning("• **Anticoagulant Use:** +35 points")
                if res.get('age', 0) > 65:
                    st.info(f"• **Age ({res.get('age')}):** +10 points (Geriatric Risk)")
                
                if not (res.get('inr', 0) > 3.5 or res.get('anticoag') or res.get('age', 0) > 65):
                    st.success("No major bleeding risk factors identified.")

            # --- TAB 2: KIDNEY (AKI) RISK ---
            with tab2:
                st.write("### Factors driving AKI Risk:")
                if res.get('diuretic'):
                    st.warning("• **Diuretic Use:** +30 points (Volume depletion risk)")
                if res.get('acei'):
                    st.warning("• **ACEi/ARB Use:** +40 points (Renal perfusion risk)")
                if res.get('sys_bp', 0) < 90:
                    st.error(f"• **Hypotension (SBP {res.get('sys_bp', 0)}):** +20 points (Perfusion threat)")
                
                if not (res.get('diuretic') or res.get('acei') or res.get('sys_bp', 0) < 90):
                    st.success("No major nephrotoxic risks identified.")

            # --- TAB 3: SEPSIS (qSOFA) ---
            with tab3:
                st.write("### Sepsis Assessment (qSOFA Criteria)")
                st.caption("Score ≥ 2 indicates high risk of poor outcome due to sepsis.")
                
                sepsis_points = 0
                if res.get('resp_rate', 0) >= 22:
                    st.error(f"• **Respiratory Rate ({res.get('resp_rate')}):** +1 Point (Target < 22)")
                    sepsis_points += 1
                if res.get('sys_bp', 0) <= 100:
                    st.error(f"• **Systolic BP ({res.get('sys_bp')}):** +1 Point (Target > 100)")
                    sepsis_points += 1
                if res.get('altered_mental'):
                    st.error("• **Altered Mental Status:** +1 Point")
                    sepsis_points += 1
                
                if sepsis_points == 0:
                    st.success("Patient meets 0 qSOFA criteria (Low Sepsis Risk).")

            # --- TAB 4: HEMODYNAMICS ---
            with tab4:
                st.write("### Hemodynamic Stability Checks")
                
                current_map = int(res.get('map_val', 0))
                current_si = res.get('shock_index', 0)

                st.write(f"**Mean Arterial Pressure (MAP):** {current_map} mmHg")
                if current_map < 65:
                    st.error("• MAP < 65 mmHg: Critical organ perfusion threat.")
                else:
                    st.success("• MAP > 65 mmHg: Perfusion is adequate.")
                    
                st.divider()
                
                st.write(f"**Shock Index (HR/SBP):** {current_si:.2f}")
                if current_si > 0.9:
                    st.error("• Index > 0.9: High probability of hidden bleeding or shock.")
                elif current_si > 0.7:
                    st.warning("• Index 0.7 - 0.9: Monitor closely for compensation.")
                else:
                    st.success("• Index < 0.7: Hemodynamically stable.")
       
       # --- CLINICAL ALERTS & PROTOCOL SUGGESTIONS (HIGHS & LOWS) ---
        st.markdown("### ⚠️ Clinical Alerts & AI Recommendations")
        violations = 0 
        
        # --- A. AIRWAY & BREATHING ---
        # 1. SpO2 (Hypoxia)
        if res.get('o2_sat', 0) > 0 and res.get('o2_sat', 0) < 88: 
            st.error(f"🚨 CRITICAL HYPOXIA (SpO2 {res['o2_sat']}%)")
            st.info("👉 **Protocol:** 15L O2 via Non-Rebreather. Prepare for RSI/Intubation. Check ABG.")
            violations += 1
        elif res.get('o2_sat', 0) > 0 and res.get('o2_sat', 0) < 92:
            st.warning(f"⚠️ Hypoxia (SpO2 {res['o2_sat']}%)")
            st.caption("👉 Suggestion: Titrate O2 to keep sats > 94%. Consider Nasal Cannula 2-4L.")
            violations += 1
        
        # 2. Respiratory Rate (High & Low)
        if res.get('resp_rate', 0) > 30:
            st.error(f"🚨 SEVERE TACHYPNEA (RR {res['resp_rate']})")
            st.info("👉 **Action:** Assess for Respiratory Distress/Acidosis. Order CXR + ABG. Rule out PE.")
            violations += 1
        elif res.get('resp_rate', 0) < 8 and res.get('resp_rate', 0) > 0:
            st.error(f"🚨 RESPIRATORY DEPRESSION (RR {res['resp_rate']})")
            st.info("👉 **Action:** Sternum rub. Check for Opioid overdose (Give Naloxone 0.4mg). Bag-Valve-Mask ventilation.")
            violations += 1

        # --- B. CIRCULATION ---
        # 3. Blood Pressure (High & Low)
        if res.get('sys_bp', 0) > 180 or res.get('dia_bp', 0) > 120: 
            st.error(f"🚨 HYPERTENSIVE CRISIS (BP {res['sys_bp']}/{res['dia_bp']})")
            st.info("👉 **Protocol:** IV Labetalol 10-20mg or Nicardipine drip. CT Head to rule out bleed.")
            violations += 1
        elif res.get('sys_bp', 0) > 0 and res.get('sys_bp', 0) < 90: 
            st.error(f"🚨 SHOCK / HYPOTENSION (BP {res['sys_bp']}/{res['dia_bp']})")
            st.info("👉 **Protocol:** Trendelenburg position. 500mL Fluid Bolus. Start Norepinephrine if MAP < 65.")
            violations += 1
            
        # 4. Heart Rate (High & Low)
        if res.get('hr', 0) > 130:
            st.error(f"🚨 SEVERE TACHYCARDIA (HR {res['hr']})")
            st.info("👉 **Action:** 12-Lead EKG STAT. Treat Pain/Fever/Sepsis. Vagal maneuvers if stable SVT.")
            violations += 1
        elif res.get('hr', 0) > 0 and res.get('hr', 0) < 40:
            st.error(f"🚨 SEVERE BRADYCARDIA (HR {res['hr']})")
            st.info("👉 **Action:** Transcutaneous Pacing pads applied. Atropine 0.5-1mg IV. Check Electrolytes.")
            violations += 1

        # --- C. DISABILITY / EXPOSURE ---
        # 5. Temperature (High & Low)
        if res.get('temp_c', 0) > 39.0:
             st.error(f"🚨 HIGH FEVER ({res['temp_c']*9/5+32:.1f}°F)")
             st.info("👉 **Action:** Blood Cultures x2. Start Antipyretics (Tylenol). Surface cooling measures.")
             violations += 1
        elif res.get('temp_c', 0) < 35.0 and res.get('temp_c', 0) > 0:
             st.error(f"🚨 HYPOTHERMIA ({res['temp_c']*9/5+32:.1f}°F)")
             st.info("👉 **Protocol:** Bear Hugger (Warm air blanket). Warm IV fluids. Monitor cardiac rhythm.")
             violations += 1
             
        # --- D. CRITICAL LABS (High & Low) ---
        # --- Glucose ---
        if res.get('glucose', 0) > 400:
            st.error(f"🚨 SEVERE HYPERGLYCEMIA ({res['glucose']} mg/dL)")
            st.info("👉 **Protocol:** IV Fluids + Insulin Drip. Check Ketones (DKA) and Osmolality (HHS).")
            violations += 1
        elif res.get('glucose', 0) > 180:
            st.warning(f"⚠️ HYPERGLYCEMIA ({res['glucose']} mg/dL)")
            st.caption("👉 **Action:** Sliding Scale Insulin coverage. Check HgbA1c. Monitor for infection (stress hyperglycemia).")
            violations += 1
        elif res.get('glucose', 0) > 0 and res.get('glucose', 0) < 70:
            st.error(f"🚨 HYPOGLYCEMIA ({res['glucose']} mg/dL)")
            st.info("👉 **Protocol:** D50 IV Push or Glucagon IM immediately. Recheck glucose in 15 mins.")
            violations += 1

        # 7. Electrolytes (Potassium)
        if res.get('potassium', 0) > 6.0:
            st.error(f"🚨 CRITICAL HYPERKALEMIA (K+ {res['potassium']})")
            st.info("👉 **Protocol:** Calcium Gluconate (Heart protect) + Insulin/D50 (Shift K+) + Albuterol.")
            violations += 1
        elif res.get('potassium', 0) > 0 and res.get('potassium', 0) < 2.5:
            st.error(f"🚨 CRITICAL HYPOKALEMIA (K+ {res['potassium']})")
            st.info("👉 **Protocol:** Urgent IV Potassium replacement (max 10mEq/hr). Continuous EKG monitoring.")
            violations += 1
            
        # 8. Hematology 
        # --- Hemoglobin (Hgb) ---
        if res.get('hgb', 0) > 0 and res.get('hgb', 0) < 7.0:
             st.error(f"🚨 CRITICAL ANEMIA (Hgb {res['hgb']} g/dL)")
             st.info("👉 **Action:** Type & Crossmatch. Transfuse 1 Unit PRBCs. Check for GI Bleed.")
             violations += 1
        elif res.get('hgb', 0) > 18.0:
             st.warning(f"⚠️ POLYCYTHEMIA (Hgb {res['hgb']} g/dL)")
             st.caption("👉 **Risk:** High stroke/clotting risk. Consider hydration or therapeutic phlebotomy.")
             violations += 1

        # --- White Blood Cells (WBC) ---
        if res.get('wbc', 0) > 0 and res.get('wbc', 0) < 1.0:
             st.error(f"🚨 SEVERE NEUTROPENIA (WBC {res['wbc']})")
             st.info("👉 **Protocol:** Neutropenic Precautions (Mask/Isolation). Start Broad-Spectrum Abx (Cefepime/Meropenem).")
             violations += 1
        elif res.get('wbc', 0) > 20.0:
             st.error(f"🚨 LEUKOCYTOSIS / INFECTION (WBC {res['wbc']})")
             st.info("👉 **Action:** Sepsis Workup (Lactate, Cultures). Rule out Leukemia or Steroid effect.")
             violations += 1

        # --- Platelets (Plt) ---
        if res.get('platelets', 0) > 0 and res.get('platelets', 0) < 20:
             st.error(f"🚨 CRITICAL THROMBOCYTOPENIA (Plt {res['platelets']})")
             st.info("👉 **Action:** Bleeding Precautions (No needles/falls). Transfuse Platelets if active bleeding.")
             violations += 1
        elif res.get('platelets', 0) > 1000:
             st.warning(f"⚠️ THROMBOCYTOSIS (Plt {res['platelets']})")
             st.caption("👉 **Risk:** Microvascular clotting. Administer Aspirin if not contraindicated.")
             violations += 1
            
        # --- E. METABOLIC, RENAL & ACID-BASE ---
        if res.get('lactate', 0) > 4.0:
             st.error(f"🚨 SEVERE LACTIC ACIDOSIS ({res['lactate']} mmol/L)")
             st.info("👉 **Protocol:** Assess for Sepsis or Ischemia. Start 30mL/kg Fluid Resuscitation. Check pH.")
             violations += 1
        elif res.get('lactate', 0) >= 2.0:
             st.warning(f"⚠️ Elevated Lactate ({res['lactate']} mmol/L)")
             st.caption("👉 **Warning:** Early sign of tissue hypoperfusion or sepsis. Repeat level in 2 hours.")
             violations += 1

        if res.get('glucose', 0) > 250 and res.get('hba1c_high', False):
             st.warning(f"⚠️ DKA RISK / METABOLIC ACIDOSIS (Glu {res['glucose']})")
             st.caption("👉 **Action:** Check Urine Ketones and Anion Gap. If positive, start DKA protocol.")
             violations += 1

        if res.get('resp_rate', 0) > 0 and res.get('resp_rate', 0) < 8:
             st.error("🚨 RESPIRATORY ACIDOSIS RISK (CO2 Retention)")
             st.info("👉 **Pathophysiology:** Patient is not breathing enough to blow off CO2. pH is likely dropping.")

        if res.get('bun', 0) > 40:
             st.warning(f"⚠️ UREMIA / HIGH BUN ({res['bun']} mg/dL)")
             st.caption("👉 **Action:** Check for GI Bleed (digested blood increases BUN) or Dehydration.")
             violations += 1

        if res.get('map_val', 0) > 0 and res.get('map_val', 0) < 65:
             st.error(f"🚨 CRITICAL LOW MAP ({int(res.get('map_val', 0))} mmHg)")
             st.info("👉 **Protocol:** Titrate Vasopressors (Levophed) to keep MAP > 65 to prevent organ failure.")
             violations += 1
            
        # --- F. PREDICTIVE MODELS ---
        if res.get('aki_risk', 0) >= 50:
             st.error(f"🚨 HIGH AKI RISK ({res['aki_risk']}%)")
             st.info("👉 **Nephrology Protocol:** 1. STOP Nephrotoxins (NSAIDs, ACEi/ARB). 2. Monitor Urine Output. 3. Avoid Contrast.")
             violations += 1
        elif res.get('aki_risk', 0) >= 20:
             st.warning(f"⚠️ Elevated AKI heuristic ({res['aki_risk']}/100)")
             st.caption("👉 **Action:** Hydrate patient. Re-check Creatinine in 12 hours.")
             violations += 1

        if res.get('bleeding_risk', 0) >= 50:
             st.error(f"🚨 CRITICAL BLEEDING RISK ({res['bleeding_risk']:.1f}%)")
             st.info("👉 **Hemorrhage Protocol:** 1. Hold Anticoagulants. 2. Type & Screen. 3. Monitor for GI Bleed.")
             violations += 1
        elif res.get('bleeding_risk', 0) >= 20:
             st.warning(f"⚠️ Moderate Bleeding Risk ({res['bleeding_risk']:.1f}%)")
             st.caption("👉 **Suggestion:** Re-evaluate anticoagulation benefit vs risk.")
             violations += 1

        if res.get('sepsis_risk', 0) >= 2:
             st.error(f"🚨 qSOFA ALERT (Clinical review required)")
             st.info("👉 **Sepsis Bundle:** Lactate -> Blood Cx -> Antibiotics -> Fluids. Time is critical.")
             violations += 1

        if res.get('hypo_risk', 0) >= 50:
             st.warning(f"⚠️ HIGH HYPOGLYCEMIA HEURISTIC ({res['hypo_risk']}/100)")
             st.caption("👉 **Action:** Patient Factors (Insulin + Renal) suggest drop is imminent. Check sugar q4h.")
             violations += 1

        if res.get('sirs_score', 0) >= 2:
             st.warning(f"⚠️ SIRS CRITERIA MET (Score {res['sirs_score']}/4)")
             st.caption("👉 **Clinical Context:** Systemic Inflammation detected. Screen for Infection (Sepsis), Trauma, or Pancreatitis.")
             violations += 1

        pp = res.get('pulse_pressure', 40)
        if pp > 60:
             st.warning(f"⚠️ WIDENED PULSE PRESSURE ({int(pp)} mmHg)")
             st.caption("👉 **Differential:** Aortic Regurgitation, Thyrotoxicosis, or ICP (Cushing's Triad).")
             violations += 1
        elif pp < 25 and pp > 0:
             st.error(f"🚨 NARROW PULSE PRESSURE ({int(pp)} mmHg)")
             st.info("👉 **Action:** Sign of Low Cardiac Output (Tamponade/Heart Failure) or Hypovolemia.")
             violations += 1
        
        # --- G. DATA INTEGRITY CHECK ---
        has_demographics = (
            res.get('age', 0) > 0 or 
            res.get('weight', 0) > 0 or 
            res.get('bmi', 0) > 0 
        )
        has_vitals = (
            res.get('sys_bp', 0) > 0 or 
            res.get('hr', 0) > 0 or 
            res.get('o2_sat', 0) > 0 or
            res.get('glucose', 0) > 0 or
            res.get('temp_c', 0) > 0
        )

        if violations == 0:
            if not has_demographics and not has_vitals:
                st.warning("⚠️ **No Data Entered:** Please input patient data to run analysis.")
            elif has_demographics and not has_vitals:
                st.warning("⚠️ **Missing Vitals:** Demographics recorded, but Vital Signs (BP, HR, SpO2) are required to determine stability.")
            else:
                st.success("✅ **Patient Stable:** No immediate life-threatening protocol violations detected.")
        
        st.divider()
        c_ai, c_txt = st.columns([1, 3])
        with c_ai:
            st.markdown("#### 🤖 AI Assessment")
            if st.button("⚡ Consult AI"):
                with st.spinner("Thinking..."):
                    ai_context = {
                        'age': res['age'], 'sbp': res['sys_bp'], 
                        'bleeding_risk': res['bleeding_risk'], 'aki_risk': res['aki_risk'],
                        'shock_index': res['shock_index']
                    }
                    response = bk.consult_ai_doctor("risk_assessment", "", ai_context)
                    st.session_state['ai_result'] = response
        
        with c_txt:
            if 'ai_result' in st.session_state:
                st.info(st.session_state['ai_result'])
            else:
                st.info("👈 Fill out the patient data form above and click 'Run Clinical Analysis' to see results.")
        
# --- MODULE 2: PATIENT HISTORY---
def render_history_sql():
    st.subheader("🗄️ Patient History Database")
    
    # Fetch Data
    df = bk.fetch_history()
    
    if not df.empty:
        # 1. Format the DataFrame for Display
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')

        # 2. Styling Helper
        def highlight_risk(val):
            if isinstance(val, (int, float)):
                if val > 50: return 'background-color: #ffcdd2; color: black;' # Red
                if val > 20: return 'background-color: #fff9c4; color: black;' # Yellow
            return ''

        # 3. Configure Columns
        st.dataframe(
            df,
            column_config={
                "timestamp": st.column_config.TextColumn("📅 Date & Time", width="medium"),
                "age": st.column_config.NumberColumn("👤 Age", format="%d yrs"),
                "gender": st.column_config.TextColumn("⚧ Gender", width="small"),
                "sbp": st.column_config.NumberColumn("❤️ SBP", format="%d mmHg"),
                "aki_risk_score": st.column_config.ProgressColumn(
                    "💧 AKI Risk", 
                    format="%d%%", 
                    min_value=0, 
                    max_value=100,
                    help="Acute Kidney Injury Probability"
                ),
                "bleeding_risk_score": st.column_config.ProgressColumn(
                    "🩸 Bleed Risk", 
                    format="%.1f%%", 
                    min_value=0, 
                    max_value=100,
                    help="Hemorrhage Risk Score"
                ),
                "status": st.column_config.TextColumn("🏥 Status"),
            },
            use_container_width=True,
            height=400,
            hide_index=True
        )

        st.divider()
        
        # 4. Analytics Section
        st.markdown("### 📈 Cohort Analytics")
        c1, c2 = st.columns(2)
        
        with c1:
            st.caption("Risk Distribution by Age")
            st.scatter_chart(df, x='age', y='bleeding_risk_score', color='status', height=250)
            
        with c2:
            st.caption("Average Vitals Trend")
            if len(df) > 1:
                st.line_chart(df.set_index('timestamp')['sbp'], height=250)
            else:
                st.info("Need more data for trend analysis.")

        # 5. Admin Actions
        with st.expander("⚙️ Database Management"):
            if st.button("🗑️ Clear All Records", type="secondary"):
                bk.clear_history()
                st.rerun()
    else:
        st.info("📭 Database is empty. Run a Risk Analysis to create records.")

# --- MODULE 3: LIVE DASHBOARD (LINKED TO CALCULATOR) ---
def render_dashboard():
    # 1. GET DATA FROM SESSION STATE
    data = st.session_state.get('patient_data', {})
    
    # Default values if no analysis has been run yet
    if not data:
        st.warning("⚠️ No patient data found. Please run the Risk Calculator first.")
        return

    is_critical = data.get('status') == 'Critical'
    
    # --- HEADER & AI BUTTON ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader(f"🛏️ Bedside Monitor: {data.get('id', 'Unknown')}")
        st.caption(f"Status: **{data.get('status', 'Unknown')}**")
    
    with c2:
        # AI DISCHARGE SUMMARY
        if st.button("✨ Generate Discharge Note", type="primary"):
            with st.spinner("Consulting Gemini 2.0..."):
                ai_summary = bk.generate_discharge_summary(data)
                st.session_state['latest_discharge_note'] = ai_summary
        
        # --- Timestamped Download ---
        if 'latest_discharge_note' in st.session_state:
            st.download_button(
                label="📥 Download Note",
                data=st.session_state['latest_discharge_note'],
                file_name=f"discharge_{data.get('id')}_{get_timestamp()}.txt",
                mime="text/plain"
            )
            
    # --- PREVIEW AREA (View Generated Summary) ---
    if 'latest_discharge_note' in st.session_state:
        with st.expander("📄 View Generated Summary", expanded=True):
            st.text_area("Edit before downloading:", value=st.session_state['latest_discharge_note'], height=200)

    st.divider()

    # --- REAL-TIME VITALS PANEL (Uses Real Inputs) ---
    with st.container(border=True):
        st.markdown("#### 📉 Real-Time Telemetry")
        
        st.caption("ℹ️ Note: Telemetry trace below is simulated based on static input data for UI demonstration.")
        
        col_chart, col_vitals = st.columns([3, 1])
        
        with col_chart:
            # Simulate a live trace based on the INPUT BP and HR
            base_sbp = data.get('sys_bp', 120)
            base_hr = data.get('hr', 80)
            
            chart_data = pd.DataFrame({
                'Time': range(20),
                'Systolic BP': np.random.normal(base_sbp, 2, 20),
                'Heart Rate': np.random.normal(base_hr, 2, 20)    
            }).melt('Time', var_name='Metric', value_name='Value')
            
            c = alt.Chart(chart_data).mark_line(interpolate='basis', strokeWidth=3).encode(
                x=alt.X('Time', axis=None),
                y=alt.Y('Value', scale=alt.Scale(zero=False)),
                color=alt.Color('Metric', scale=alt.Scale(range=['#FF4B4B', '#00CC96']))
            ).properties(height=200)
            
            st.altair_chart(c, use_container_width=True)

        with col_vitals:
            st.markdown(f"""
            <div style="background-color:#0E1117; padding:15px; border-radius:10px; text-align:center; border: 1px solid #333;">
                <h3 style="color:#FF4B4B; margin:0;">{int(data.get('sys_bp', 0))}</h3>
                <p style="color:gray; font-size:12px; margin:0;">mmHg (SBP)</p>
                <hr style="margin: 10px 0; border-color:#333;">
                <h3 style="color:#00CC96; margin:0;">{int(data.get('hr', 0))}</h3>
                <p style="color:gray; font-size:12px; margin:0;">BPM (HR)</p>
                <hr style="margin: 10px 0; border-color:#333;">
                <h3 style="color:#00A6ED; margin:0;">{int(data.get('o2_sat', 0))}%</h3>
                <p style="color:gray; font-size:12px; margin:0;">SpO2</p>
            </div>
            """, unsafe_allow_html=True)

    # --- RISK METRICS (From Analysis) ---
    st.markdown("#### ⚠️ Risk Stratification")
    r1, r2, r3, r4 = st.columns(4)
    
    r1.metric("🩸 Bleeding model estimate", f"{data.get('bleeding_risk', 0):.1f}%",
              "High" if data.get('bleeding_risk', 0) > 50 else "Normal", delta_color="inverse",
              help="Probability of major hemorrhage based on XGBoost model.")
    
    r2.metric("💧 AKI Risk", f"{data.get('aki_risk', 0)}%", 
              "Critical" if data.get('aki_risk', 0) > 50 else "Normal", delta_color="inverse",
              help="Acute Kidney Injury Risk based on KDIGO criteria.")
    
    r3.metric("🦠 Sepsis Score", f"{data.get('sepsis_risk', 0)}", 
              "Alert" if data.get('sepsis_risk', 0) >= 2 else "Normal", delta_color="inverse",
              help="qSOFA Score (0-3). ≥2 indicates high sepsis risk.")
    
    r4.metric("🌡️ Temp", f"{data.get('temp_c', 37.0)*9/5+32:.1f}°F", "Fever" if data.get('temp_c', 37) > 38 else "Normal", delta_color="inverse")

# --- MODULE 4: BATCH ANALYSIS (SMART VALIDATION & NEWS-2) ---
def render_batch_analysis():
    st.subheader("Bulk Patient Processing & Diagnostic Triage")
    
    # 1. Helper: Download Template
    with st.expander("ℹ️  Download CSV Template"):
        sample_data = {
            'Age': [65, 72], 'Gender': ['Male', 'Female'], 'Weight_kg': [80, 65],
            'Systolic_BP': [130, 90], 'Diastolic_BP': [80, 50], 'Heart_Rate': [72, 110],
            'Resp_Rate': [16, 24], 'Temp_C': [37.0, 38.5], 'O2_Sat': [98, 92],
            'WBC': [6.0, 15.0], 'Glucose': [110, 5.5], 'Creatinine': [1.1, 150],
            'INR': [1.0, 1.2], 'Altered_Mental': [0, 1], 'Anticoagulant': [1, 0],
            'Heart_Failure': [0, 1], 'Liver_Disease': [0, 0], 'Hx_GI_Bleed': [0, 0]
        }
        df_sample = pd.DataFrame(sample_data)
        csv_template = df_sample.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Template", csv_template, "patient_data.csv", "text/csv")

    # 2. CSV PROCESSOR
    st.markdown("#### 📤 Upload Patient Batch")
    uploaded_csv = st.file_uploader("Upload Patient Data (CSV)", type=["csv"])
    
    if uploaded_csv:
        try:
            raw_df = pd.read_csv(uploaded_csv)
            
            # --- A. SMART MAPPING ---
            col_map = {
                'sbp':'Systolic_BP', 'sys':'Systolic_BP', 'hr':'Heart_Rate', 'rr':'Resp_Rate', 'temp':'Temp_C',
                'spo2':'O2_Sat', 'cr':'Creatinine', 'wbc':'WBC', 'glu':'Glucose', 'sugar':'Glucose'
            }
            df = raw_df.rename(columns=lambda x: col_map.get(x.lower(), x))
            
            # Fill missing with safe defaults for calculation
            req_cols = ['Age','Systolic_BP','Heart_Rate','Resp_Rate','Temp_C','O2_Sat','Creatinine','Glucose']
            for c in req_cols:
                if c not in df.columns: df[c] = 0

            if st.button("⚡ Run Clinical Analysis", type="primary"):
                with st.spinner("Validating Data & Calculating Risks..."):
                    
                    # --- B. SMART UNIT CONVERSION & VALIDATION ---
                    def clean_row(row):
                        # 1. Glucose: Convert mmol/L to mg/dL
                        if 0 < row['Glucose'] < 30:
                            row['Glucose'] = row['Glucose'] * 18
                        
                        # 2. Creatinine: Convert umol/L to mg/dL
                        if row['Creatinine'] > 20:
                            row['Creatinine'] = row['Creatinine'] / 88.4
                            
                        return row
                    
                    df = df.apply(clean_row, axis=1)

                    # --- C. ADVANCED SCORING (NEWS-2) ---
                    def calculate_news(row):
                        score = 0
                        # Resp Rate
                        if row['Resp_Rate'] <= 8 or row['Resp_Rate'] >= 25: score += 3
                        elif row['Resp_Rate'] >= 21: score += 2
                        elif row['Resp_Rate'] <= 11: score += 1
                        
                        # SpO2
                        if row['O2_Sat'] <= 91: score += 3
                        elif row['O2_Sat'] <= 93: score += 2
                        elif row['O2_Sat'] <= 95: score += 1
                        
                        # Systolic BP
                        if row['Systolic_BP'] <= 90 or row['Systolic_BP'] >= 220: score += 3
                        elif row['Systolic_BP'] <= 100: score += 2
                        elif row['Systolic_BP'] <= 110: score += 1
                        
                        # Heart Rate
                        if row['Heart_Rate'] <= 40 or row['Heart_Rate'] >= 131: score += 3
                        elif row['Heart_Rate'] >= 111: score += 2
                        elif row['Heart_Rate'] <= 50 or row['Heart_Rate'] >= 91: score += 1
                        
                        # Consciousness
                        if row.get('Altered_Mental', 0) == 1: score += 3
                        
                        # Temp
                        if row['Temp_C'] <= 35.0: score += 3
                        elif row['Temp_C'] >= 39.1: score += 2
                        elif row['Temp_C'] <= 36.0 or row['Temp_C'] >= 38.1: score += 1
                        
                        return score

                    df['NEWS_Score'] = df.apply(calculate_news, axis=1)

                    # --- D. DIAGNOSTIC LABELS ---
                    def get_status(row):
                        alerts = []
                        # 1. NEWS-2 Interpretation
                        if row['NEWS_Score'] >= 7: alerts.append("CRITICAL (NEWS ≥7)")
                        elif row['NEWS_Score'] >= 5: alerts.append("Urgent (NEWS ≥5)")
                        
                        # 2. Specific Organ Failures
                        if row['Systolic_BP'] > 180: alerts.append("HTN Crisis")
                        if row['Creatinine'] > 2.0: alerts.append("AKI Warning")
                        if row.get('WBC', 0) > 12: alerts.append("Leukocytosis")
                        
                        return " + ".join(alerts) if alerts else "Stable"

                    df['Clinical_Status'] = df.apply(get_status, axis=1)
                    
                    # Run AI Prediction (Create simple input df for the XGBoost model)
                    model_inputs = pd.DataFrame()
                    model_inputs['age'] = df.get('Age', 0)
                    model_inputs['inr'] = df.get('INR', 1.0)
                    model_inputs['anticoagulant'] = df.get('Anticoagulant', 0)
                    model_inputs['gi_bleed'] = df.get('Hx_GI_Bleed', 0)
                    model_inputs['high_bp'] = df['Systolic_BP'].apply(lambda x: 1 if x > 140 else 0)
                    model_inputs['antiplatelet'] = 0
                    model_inputs['gender_female'] = 0 
                    model_inputs['weight'] = df.get('Weight_kg', 70)
                    model_inputs['liver_disease'] = df.get('Liver_Disease', 0)
                    
                    df['Bleed_Risk_%'] = bleeding_model.predict(model_inputs)

                    # --- E. DISPLAY ---
                    def color_rows(val):
                        if 'CRITICAL' in str(val): return 'background-color: #ffcdd2; color: black; font-weight: bold;'
                        if 'Urgent' in str(val) or 'Warning' in str(val): return 'background-color: #fff3cd; color: black;'
                        return ''

                    st.success(f"Processed {len(df)} records with enhanced validation.")
                    st.dataframe(
                        df[['Clinical_Status', 'NEWS_Score', 'Age', 'Systolic_BP', 'Bleed_Risk_%']]
                        .style.map(color_rows, subset=['Clinical_Status']), 
                        use_container_width=True
                    )
                    
                    # ---Timestamped Download ---
                    csv_result = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Analyzed Data", 
                        csv_result, 
                        f"analyzed_patients_{get_timestamp()}.csv", 
                        "text/csv"
                    )
                
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# --- MODULE 5: MEDICATION CHECKER ---
def render_medication_checker():
    st.subheader("💊 Drug-Drug Interaction Checker")
    st.caption("Checks for Critical and Major interactions from backend database + AI Analysis.")
    
    col_d1, col_d2 = st.columns(2)
    d1 = col_d1.text_input("Drug A", placeholder="e.g. Warfarin")
    d2 = col_d2.text_input("Drug B", placeholder="e.g. Ibuprofen")

    if d1 and d2:
        st.divider()
        
        # 1. Database Check (Deterministic)
        res = bk.check_interaction(d1, d2)
        
        if res:
            if "CRITICAL" in res: 
                st.error(f"❌ {res}")
            elif "MAJOR" in res: 
                st.warning(f"⚠️ {res}")
            elif "MODERATE" in res: 
                st.info(f"ℹ️ {res}")
        else: 
            st.warning(f"⚠️ **{d1}** + **{d2}** not found in the high-alert database.")
            st.markdown("👉 **Recommendation:** This does not guarantee safety. Please use the **AI Pharmacist** below for a comprehensive check.")

        # 2. AI Analysis Button
        st.divider()
        st.markdown("#### 🧠 AI Pharmacist Analysis")
        st.caption("Get a detailed explanation of mechanism, management, and safety profile.")
        
        if st.button("⚡ Analyze Interaction with AI"):
            with st.spinner("Consulting AI Pharmacist..."):
                ai_report = bk.analyze_drug_interactions([d1, d2])
                st.markdown(ai_report)
                st.caption("⚠️ AI-Generated. Verify with standard drug compendiums.")

# --- MODULE 6: CHATBOT ---
def render_chatbot():
    st.subheader("AI Clinical Assistant")
    st.caption("Database covers 250+ clinical topics (Cardio, Resp, Neuro, Pharm, Labs).")
    
    q = st.text_input("Ask a clinical question:")
    if q:
        with st.chat_message("assistant"):
            st.write(bk.chatbot_response(q))

# --- MODULE 7: AI DIAGNOSTICIAN ---
def render_ai_diagnostician():
    st.subheader("🧠 AI-Powered Clinical Consultant")
    st.caption("Ask complex clinical questions or simulate differential diagnoses.")
    
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I am your AI Clinical Consultant. Describe a patient case or ask a medical question."}]

    # Display History
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Type your clinical query here..."):
        # 1. User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # 2. AI Response
        with st.spinner("Thinking..."):
            # Get context from session if available
            current_data = st.session_state.get('patient_data', {})
            
            # Call Backend
            response = bk.consult_ai_doctor("provider", prompt, current_data)
            
            # 3. Display AI Message
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

# ---------------------------------------------------------
# 3. MAIN APP CONTROLLER
# ---------------------------------------------------------
if not st.session_state['entered_app']:
    render_cover_page()
else:
    with st.sidebar:
        st.title("Navigation")
        menu = st.radio("Select Module", [
            "Hospital Workspace",
            "Risk Calculator", 
            "Patient History (SQL)",
            "Live Dashboard", 
            "Batch Analysis (CSV)", 
            "Medication Checker", 
            "📚 Medical Glossary",
            "🧠 AI Clinical Consultant"
        ])
        st.info("v5.0 • Admission to discharge")

    if menu == "Hospital Workspace":
        render_hospital(bleeding_model)
    elif menu == "Risk Calculator":
        render_risk_calculator()
    elif menu == "Patient History (SQL)":
        render_history_sql()
    elif menu == "Live Dashboard":
        render_dashboard()
    elif menu == "Batch Analysis (CSV)":
        render_batch_analysis()
    elif menu == "Medication Checker":
        render_medication_checker()
    elif menu == "📚 Medical Glossary":
        render_chatbot()
    elif menu == "🧠 AI Clinical Consultant":
        render_ai_diagnostician()
      # ---------------------------------------------------------
    # DEBUG: ADMIN PANEL (Only visible if URL has ?admin=true)
    # ---------------------------------------------------------
    # Check if the URL is https://.../?admin=true
    # Model retraining is disabled in the hosted demo.
