from dataclasses import replace


def annotate(llm, text, spans, enabled):
    rows = llm.annotate(text, spans, enabled)
    if not isinstance(rows, list) or len(rows) != len(spans):
        raise ValueError('annotation must cover each span exactly once')
    ids = {s.id for s in spans}
    by_id = {}
    for row in rows:
        if not isinstance(row, dict) or row.get('id') not in ids or row['id'] in by_id:
            raise ValueError('invalid annotation id')
        if row.get('kind') not in {'ASSERTION', 'CONDITION'} or row.get('scope') not in {'NONE', 'LOCAL', 'FORWARD', 'GLOBAL'}:
            raise ValueError('invalid annotation type or scope')
        target = row.get('modifies_id')
        if target is not None and (target not in ids or target == row['id']):
            raise ValueError('invalid modification target')
        pre = row.get('preconditions', [])
        if not isinstance(pre, list) or any(not isinstance(p, str) or not p.strip() for p in pre):
            raise ValueError('invalid preconditions')
        if enabled and row['kind'] == 'ASSERTION' and not 1 <= len(pre) <= 2:
            raise ValueError('assertion requires one or two preconditions')
        by_id[row['id']] = row
    return [replace(s, kind=by_id[s.id]['kind'], scope=by_id[s.id]['scope'], modifies_id=by_id[s.id].get('modifies_id'), preconditions=tuple(by_id[s.id].get('preconditions', [])) if enabled else ()) for s in spans]
