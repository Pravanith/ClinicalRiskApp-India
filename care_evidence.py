"""Source-linked draft changes; evidence is context, not clinical verification."""
import re
from care_logic import intake_from_text
from voice_entry import FIELDS


def evidence_rows(text, before, after):
    # Split sentence endings without splitting decimal measurements.
    phrases = [s.strip() for s in re.split(r'(?<=[.!?])\s+|[;\n]+', text) if s.strip()]
    parsed = [(s, intake_from_text(s, before)[0]) for s in phrases]
    rows = []

    def add(label, old, new, supports):
        sources = [s for s, candidate in parsed if supports(candidate)]
        rows.append({'Field': label, 'Before': str(old), 'Draft': str(new),
                     'Review': 'Changed existing value' if old != 'Not recorded' else 'New entry',
                     'Transcript evidence': '\n'.join(sources) if sources else text,
                     'Evidence scope': 'Matching phrase' if sources else 'Full note — review context'})

    for key, value in after.get('observations', {}).items():
        old = before.get('observations', {}).get(key, 'Not recorded')
        if old != value:
            add('Temperature °F' if key == 'temp_c' else FIELDS.get(key, (key.replace('_', ' ').title(),))[0],
                (round(old*9/5+32, 1) if isinstance(old, (float,int)) and old else 'Not recorded') if key == 'temp_c' else old,
                round(value*9/5+32, 1) if key == 'temp_c' else value,
                lambda candidate, k=key, v=value: candidate.get('observations', {}).get(k) == v)
    old_conditions = set(before.get('conditions', []))
    new_conditions = set(after.get('conditions', []))
    for condition in sorted(old_conditions ^ new_conditions):
        present = condition in new_conditions
        add(condition, 'Present' if condition in old_conditions else 'Not recorded',
            'Present' if present else 'Explicitly denied in note',
            lambda candidate, c=condition, p=present: (c in candidate.get('conditions', [])) == p)
    for key in ('allergies', 'reason'):
        value = after.get(key, '')
        old = before.get(key) or 'Not recorded'
        if value and value != before.get(key):
            add(key.title(), old, value, lambda candidate, k=key, v=value: candidate.get(k) == v)
    prior_ids = {m['id'] for m in before.get('medications', [])}
    for med in after.get('medications', []):
        if med['id'] not in prior_ids:
            rows.append({'Field': 'Medication: '+med['name'], 'Before': 'Not recorded',
                         'Draft': f"{med['dose']} {med['unit']} · {med['frequency']}",
                         'Review': 'Unreviewed medication candidate',
                         'Transcript evidence': med.get('source', text),
                         'Evidence scope': 'Parser medication phrase (normalized)'})
    if before.get('allergy_status') == 'No known allergies' and after.get('allergy_status') == 'Reported':
        rows.append({'Field': 'Allergy history conflict', 'Before': 'No known allergies',
                     'Draft': after.get('allergies', ''), 'Review': 'Resolve conflicting allergy histories',
                     'Transcript evidence': text, 'Evidence scope': 'Full note — review context'})
    return rows
