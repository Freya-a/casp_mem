from casp_mem.segmentation.segmenter import validate

def assemble(text, spans, patches, locked):
    if not validate(text, spans):
        raise ValueError('invalid segmentation')
    ids = {s.id for s in spans}
    by_id = {}
    for p in patches:
        if p.span_id not in ids or p.span_id in locked or p.span_id in by_id:
            raise ValueError('patch overlaps a locked, unknown or duplicate span')
        by_id[p.span_id] = p
    parts, locations, pos = [], {}, 0
    for s in spans:
        piece = by_id[s.id].text if s.id in by_id else text[s.start:s.end]
        if not isinstance(piece, str):
            raise ValueError('patch text must be a string')
        locations[s.id] = (pos, pos + len(piece))
        parts.append(piece)
        pos += len(piece)
    return ''.join(parts), locations
