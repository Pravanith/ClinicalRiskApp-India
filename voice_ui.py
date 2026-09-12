"""Speech capture, transcription and reversible form updates."""
import io
from pathlib import Path
import streamlit as st
from voice_entry import extract_fields, FIELDS
from admission import EXTRA_FIELDS

@st.cache_resource
def speech_model():
    from faster_whisper import WhisperModel
    return WhisperModel('base.en', device='cpu', compute_type='int8', download_root=str(Path(__file__).parent / '.speech_models'))

def apply_transcript():
    result = extract_fields(st.session_state.get('voice_transcript', ''))
    st.session_state['voice_previous'] = {
        key: st.session_state.get('risk_'+key, None if key in EXTRA_FIELDS else 'kg' if key=='w_unit' else 'Male' if key=='gender' else False if isinstance(value,bool) else 0.0 if isinstance(value,float) else 0)
        for key,value in result.values.items()
    }
    for key, value in result.values.items():
        st.session_state['risk_'+key] = value
    if 'temp_c' in result.values:
        st.session_state['risk_temp_f'] = result.values['temp_c']*9/5+32
    st.session_state['voice_result'] = result
    st.session_state.pop('analysis_results', None)
    st.session_state['patient_data'] = {}

def transcribe_recording():
    audio = st.session_state.get('voice_recording')
    if audio is None: return
    st.session_state.pop('voice_error', None)
    try:
        if audio.size > 20 * 1024 * 1024:
            raise ValueError('Recording is too long. Please record a shorter note (under 20 MB).')
        segments, _ = speech_model().transcribe(io.BytesIO(audio.getvalue()), language='en', vad_filter=True, condition_on_previous_text=False,
            initial_prompt='Clinical observations: age, gender, weight, height, blood pressure, heart rate, respiratory rate, temperature, oxygen saturation, creatinine, blood urea nitrogen, potassium, glucose, WBC, hemoglobin, platelets, INR, lactate. Medical history: anticoagulant use, liver disease, heart failure, history of GI bleed, NSAID use, active chemotherapy, diuretic use, ACE inhibitor, ARB, insulin, uncontrolled diabetes, altered mental status, confusion. Additional observations: pain score, Glasgow Coma Scale, oxygen flow, urine volume, urine duration, sodium, chloride, bicarbonate, calcium, magnesium, phosphate, albumin, bilirubin, ALT, AST, alkaline phosphatase, HbA1c. Answers: yes, no, unknown.')
        transcript = ' '.join(segment.text.strip() for segment in segments).strip()
        if not transcript: raise ValueError('No speech detected. Please record again or type the observations.')
        st.session_state['voice_transcript'] = transcript
        apply_transcript()
    except ImportError:
        st.session_state['voice_error'] = 'Speech dependencies are missing. Install requirements.txt, restart, and record again. Text entry is still available.'
    except Exception as exc:
        st.session_state['voice_error'] = f'Transcription did not complete. Fields were not changed. {exc}'

def undo_voice():
    for key,value in st.session_state.pop('voice_previous',{}).items(): st.session_state['risk_'+key] = value
    c = st.session_state.get('risk_temp_c', 0)
    st.session_state['risk_temp_f'] = c*9/5+32 if c else None
    st.session_state.pop('voice_result',None)
    st.session_state.pop('analysis_results',None)
    st.session_state['patient_data'] = {}

def render_voice_entry():
    with st.container(border=True):
        st.markdown('#### 🎙️ Voice-assisted patient entry')
        st.caption('Record an English clinical note, then press Stop. Transcription and field filling start automatically. Review the filled fields before running analysis.')
        st.caption('Audio is processed on the app server with Whisper, without a speech API. The first recording downloads the speech model. Microphone access requires localhost or HTTPS.')
        st.audio_input('Record patient observations', key='voice_recording', on_change=transcribe_recording)
        if st.session_state.get('voice_error'): st.error(st.session_state['voice_error'])
        st.text_area('Transcript — editable, or paste a clinical note', key='voice_transcript', height=100,
                     placeholder='Age 68. Gender female. Blood pressure 120 over 80. Heart rate 76. Temperature 98.6 Fahrenheit. Creatinine 1.2. INR 2.1. Anticoagulant yes. No liver disease.')
        a,b = st.columns(2)
        a.button('Fill fields from transcript', on_click=apply_transcript, use_container_width=True)
        b.button('Undo last fill', on_click=undo_voice, disabled=not st.session_state.get('voice_previous'), use_container_width=True)
        result = st.session_state.get('voice_result')
        if result:
            if result.values:
                st.success(f'{len(result.values)} fields matched. Unmentioned fields retain their current values.')
                history = [(k, v) for k, v in result.values.items() if isinstance(v, bool)]
                if history:
                    st.write('Medical history applied: ' + '; '.join(k.replace('_', ' ') + (' → checked' if v else ' → unchecked') for k, v in history))
                with st.expander('Review extracted values and source text'):
                    st.dataframe([{'Field': ({**FIELDS, **EXTRA_FIELDS})[k][0] if k in {**FIELDS, **EXTRA_FIELDS} else k.replace('_',' ').title(), 'Value':str(v), 'Source':result.evidence[k]} for k,v in result.values.items()], hide_index=True,use_container_width=True)
            else: st.info('No supported observations found. Say each field name followed by its value.')
            for warning in result.warnings: st.warning(warning)
        with st.expander('Supported dictation'):
            st.write('All numeric patient fields, gender, and explicitly stated yes/no history flags are supported. Say “blood pressure 120 over 80”, “creatinine one point two”, or “insulin, yes”. Commas between labels and answers are accepted. Unlabelled numbers are ignored. Conflicting or uncertain history is left unchanged for manual review. If a medical term is misheard, correct it in the transcript and click Fill fields from transcript.')
            st.write('Use the units printed on the form. Fahrenheit temperatures, pounds, and heights in inches or meters are supported. Recording ends when you press Stop; silent pauses do not stop recording.')
