Deploy this folder as a NEW repository so old database commits are not carried into it.
Streamlit entrypoint: app.py
Python: 3.11
Optional secrets: GEMINI_API_KEY and GEMINI_MODEL.
Never commit secrets or patient databases. The database is created empty at runtime.
Speech model weights download on first recording.
Public demo: fictional data only; no authenticated patient access.
The original local workspace is unchanged.

Voice review update: deploy both care_ui.py and care_evidence.py together. The intake panel now displays source-linked changes beside the transcript and an encounter overview. Evidence describes the most recent transcript fill; subsequent manual edits may differ.
