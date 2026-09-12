"""Conservative, local extraction of explicitly labelled clinical observations."""
import re
from dataclasses import dataclass, field
from admission import EXTRA_FIELDS, ZERO_VALID

# Ranges match the existing form; these are input limits, not reference ranges.
FIELDS = {
 'age': ('Age', r'age|aged', 0, 120, int),
 'weight_input': ('Weight', r'weight', 0, 400, float),
 'height': ('Height', r'height', 0, 250, int),
 'sys_bp': ('Systolic BP', r'systolic(?: blood pressure| bp)?', 0, 300, int),
 'dia_bp': ('Diastolic BP', r'diastolic(?: blood pressure| bp)?', 0, 200, int),
 'hr': ('Heart rate', r'heart rate|pulse(?: rate)?', 0, 300, int),
 'resp_rate': ('Respiratory rate', r'respiratory rate|respiration rate|resp rate|respirations', 0, 60, int),
 'temp_c': ('Temperature °C', r'temperature|temp', 0, 45, float),
 'o2_sat': ('Oxygen saturation', r'oxygen saturation|o2 sat(?:uration)?|spo2|saturation', 0, 100, int),
 'creat': ('Creatinine', r'creatinine', 0, 20, float),
 'bun': ('BUN', r'blood urea nitrogen|bun', 0, 100, int),
 'potassium': ('Potassium', r'potassium', 0, 10, float),
 'glucose': ('Glucose', r'blood glucose|blood sugar|glucose', 0, 1000, int),
 'wbc': ('WBC', r'white blood cell(?: count)?|white cell count|wbc', 0, 50, float),
 'hgb': ('Hemoglobin', r'hemoglobin|haemoglobin|hgb', 0, 20, float),
 'platelets': ('Platelets', r'platelet(?: count)?s?', 0, 1000, int),
 'inr': ('INR', r'inr|i n r', 0, 10, float),
 'lactate': ('Lactate', r'lactate', 0, 20, float),
}
FLAGS = {
 'anticoag': r'anticoagulant use|anticoagulants?', 'liver_disease': r'liver disease',
 'heart_failure': r'heart failure', 'gi_bleed': r'(?:history of )?(?:gi|g\.?\s*i\.?) bleed(?:ing)?',
 'nsaid': r'(?:nsaids?|n\.?\s*s\.?\s*a\.?\s*i\.?\s*d\.?)(?: use)?', 'active_chemo': r'active chemo(?:therapy)?',
 'diuretic': r'diuretic use|diuretics?', 'acei': r'(?:acei|ace inhibitor|ace inhibitors)(?:\s*/\s*arb)?|a\.?\s*c\.?\s*e\.? inhibitor|arb',
 'insulin': r'insulin', 'hba1c_high': r'uncontrolled diabetes',
 'altered_mental': r'altered mental status|confusion',
}
ONES = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(), range(20)))
TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20,100,10)))
WORDS = r'(?:' + '|'.join([*ONES,*TENS,'hundred','thousand','point']) + r')'

def normalize_numbers(text):
    text = re.sub(r'(?<=[a-z])-(?=[a-z])', ' ', text.lower())
    def convert(m):
        words = m.group().replace(' and ', ' ').split()
        if len(words) > 1 and words[0] in ONES and words[1] in TENS:
            return m.group()
        if 'point' in words:
            idx = words.index('point')
            left = number(words[:idx]) if idx else 0
            right = words[idx+1:]
            if not right or any(w not in ONES or ONES[w] > 9 for w in right):
                return m.group()
            return str(left) + '.' + ''.join(str(ONES[w]) for w in right)
        return str(number(words))
    def number(words):
        total = current = 0
        # "one twenty" is ambiguous; don't guess shorthand.
        for w in words:
            if w == 'hundred': current = (current or 1) * 100
            elif w == 'thousand': total += (current or 1) * 1000; current = 0
            else: current += ONES.get(w, TENS.get(w, 0))
        return total + current
    return re.sub(r'\b'+WORDS+r'(?:[ -]+(?:and[ -]+)?'+WORDS+r')*\b', convert, text)

@dataclass
class Extraction:
    values: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

def extract_fields(transcript):
    text = normalize_numbers(transcript)
    result = Extraction()
    candidates = {}
    def add(key, value, source):
        candidates.setdefault(key, []).append((value, source))
    number = r'(-?\d+(?:\.\d+)?)'
    link = r'\s*(?:(?:is|was|of|equals|at)\s*)?[:=]?\s*'
    # Match labels first so one observation cannot consume the next observation's unit.
    numeric_fields = {**FIELDS, **EXTRA_FIELDS}
    all_labels = '|'.join(f'(?:{spec[1]})' for spec in numeric_fields.values())
    for key, (_, alias, low, high, cast) in numeric_fields.items():
        for m in re.finditer(r'\b(?:'+alias+r')\b'+link+number, text):
            value = float(m.group(1))
            prefix = text[max(0,m.start()-30):m.start()]
            tail = re.split(r'\b(?:'+all_labels+r')\b|[,;\n]', text[m.end():], maxsplit=1)[0].strip()
            if re.search(r'\b(previous|prior|baseline|yesterday|not|no)\s*$', prefix):
                result.warnings.append(f'Skipped historical or negated {key}: {m.group()}'); continue
            if re.match(r'(?:to|or|[-–])\s*\d', tail):
                result.warnings.append(f'Ambiguous range for {key}; enter one value.'); continue
            if key == 'urine_volume' and re.match(r'(?:ml|milliliters?)\s*(?:/|per)\s*(?:h|hour|min)',tail):
                result.warnings.append('Urine output was given as a rate. Enter total urine volume and collection duration separately.'); continue
            if key == 'urine_hours' and re.match(r'minutes?|mins?\b',tail): value /= 60
            if key == 'temp_c' and re.match(r'(?:degrees?\s*)?(?:°\s*)?(?:f\b|fahrenheit)', tail): value = (value-32)*5/9
            elif key == 'weight_input':
                if re.match(r'lb\b|lbs\b|pounds?\b', tail): add('w_unit', 'lbs', m.group()+' '+tail)
                else: add('w_unit', 'kg', m.group())
            elif key == 'height' and re.match(r'inches|inch\b|in\b', tail): value *= 2.54
            elif key == 'height' and re.match(r'meters?\b|metres?\b|m\b', tail): value *= 100
            elif re.match(r'µmol|umol|mmol|mg/|g/|g per|mg per', tail):
                expected = {'creat': r'mg(?:/| per )d[lL]', 'glucose': r'mg(?:/| per )d[lL]', 'hgb': r'g(?:/| per )d[lL]', 'potassium': r'mmol(?:/| per )[lL]', 'lactate': r'mmol(?:/| per )[lL]'}
                expected.update({k:r'mmol(?:/| per )[lL]' for k in ('sodium','chloride','bicarbonate')})
                expected.update({k:r'mg(?:/| per )d[lL]' for k in ('calcium','magnesium','phosphate','bilirubin')})
                expected['albumin']=r'g(?:/| per )d[lL]'
                if key not in expected or not re.match(expected[key],tail):
                    result.warnings.append(f'Unsupported units for {key}; convert to the form units.'); continue
            if cast is int and value != int(value) and key != 'height':
                result.warnings.append(f'{key} requires a whole number.'); continue
            valid = low <= value <= high if key in ZERO_VALID or key == 'gcs' else low < value <= high
            if not valid:
                result.warnings.append(f'{key} is outside the supported input range: {value:g}.'); continue
            add(key, int(round(value)) if cast is int else round(value,2), m.group()+' '+tail[:30])
    for m in re.finditer(r'\b(?:blood pressure|bp)'+link+r'(\d{2,3})\s*(?:/|over)\s*(\d{2,3})\b', text):
        if re.search(r'\b(previous|prior|baseline|yesterday|not|no)\s*$', text[max(0,m.start()-30):m.start()]): continue
        sbp, dbp = map(int, m.groups())
        if 0 < dbp < sbp <= 300 and dbp <= 200:
            add('sys_bp', sbp,m.group()); add('dia_bp',dbp,m.group())
        else: result.warnings.append('Invalid blood pressure pair; review systolic and diastolic values.')
    for m in re.finditer(r'\b(?:gender|sex)'+link+r'(male|female)\b', text): add('gender',m.group(1).title(),m.group())
    # Speech recognition commonly puts punctuation between a label and its answer.
    # A following label prevents "heart failure. No liver disease" from assigning
    # liver disease's negation to heart failure.
    history_link = r'\s*(?:\([^)]{0,50}\)\s*)?(?:[,;:.!?=–—-]\s*)*(?:(?:is|was)\s*)?'
    known_history = '|'.join('(?:'+alias+')' for alias in FLAGS.values())
    answer = r'(yes|no|true|false|present|absent|unknown|unsure|uncertain)\b'
    matched_answers = []
    for key, alias in FLAGS.items():
        for m in re.finditer(r'\b(?:'+alias+r')\b'+history_link+answer,text):
            if re.match(r'\s+(?:'+known_history+r')\b', text[m.end():]):
                continue
            matched_answers.append(m.span(1))
            if m.group(1) in ('unknown','unsure','uncertain'):
                add(key,None,m.group())
            else:
                add(key,m.group(1) in ('yes','true','present'),m.group())
        for m in re.finditer(r'\b(no(?: history of)?|denies|without)\s+(?:'+alias+r')\b',text):
            matched_answers.append(m.span(1))
            add(key,False,m.group())
    for m in re.finditer(r'\b(?:yes|no|true|false|present|absent|unknown|unsure|uncertain)\b',text):
        if not any(start <= m.start() < end for start,end in matched_answers):
            context=text[max(0,m.start()-45):m.end()].strip()
            result.warnings.append(f'Unmatched history answer: "…{context}". Check the transcript spelling and field name; no checkbox was inferred from this answer.')
    for key, entries in candidates.items():
        if any(value is None for value,_ in entries):
            result.warnings.append(f'Uncertain {key} history; field unchanged. Review manually.')
            continue
        if len({v for v,_ in entries}) > 1:
            result.warnings.append(f'Conflicting {key} values; field unchanged.'); continue
        result.values[key], result.evidence[key] = entries[-1]
    if 'weight_input' not in result.values: result.values.pop('w_unit',None)
    if candidates.get('w_unit') and 'w_unit' not in result.values: result.values.pop('weight_input',None)
    return result
