from casp_mem.schemas import ScopeDecision

LABELS = {'ATTR_CONFLICT', 'CONDITION_SHIFT', 'CASCADED_INVALID', 'LOCKED_INVARIANT', 'UNCERTAIN'}

def review(llm, text, evidence, spans, selected):
    rows = llm.review(text, evidence, spans, selected)
    if not isinstance(rows, list) or len(rows) != len(spans):
        raise ValueError('review must cover every span')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'span_id', 'label', 'reason'} or row['span_id'] in result or row['label'] not in LABELS or not isinstance(row['reason'], str) or not row['reason'].strip():
            raise ValueError('invalid semantic review')
        result[row['span_id']] = ScopeDecision(**row)
    if set(result) != {s.id for s in spans}:
        raise ValueError('review ids do not cover spans')
    if any(result[i].label == 'LOCKED_INVARIANT' for i in selected):
        raise ValueError('review cannot lock selected spans')
    return [result[s.id] for s in spans]
