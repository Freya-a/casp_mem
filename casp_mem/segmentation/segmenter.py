import hashlib
import re
from dataclasses import asdict
from casp_mem.schemas import Span

# Keep delimiters and whitespace exactly where they appeared.
BOUNDARY = re.compile(r'[。！？；;!?，,.]|\b(?:if|when|unless|while|because)\b|如果|假如|除非|当|因为', re.I)

def segment(text):
    if not text:
        return []
    cuts = {0, len(text)}
    for m in BOUNDARY.finditer(text):
        cuts.add(m.end() if len(m.group()) == 1 else m.start())
    boundaries = sorted(cuts)
    spans = [Span(i, a, b) for i, (a, b) in enumerate(zip(boundaries, boundaries[1:])) if b > a]
    if not validate(text, spans):
        raise ValueError('segmentation is not lossless')
    return spans

def validate(text, spans):
    pos = 0
    ids = set()
    for s in spans:
        if type(s.id) is not int or type(s.start) is not int or type(s.end) is not int or s.id in ids or s.start != pos or not s.start < s.end <= len(text):
            return False
        ids.add(s.id)
        pos = s.end
    return pos == len(text) and ''.join(text[s.start:s.end] for s in spans) == text

def cache_record(text, spans, annotation_config=None):
    if not validate(text, spans):
        raise ValueError('invalid spans')
    return {'text_hash': hashlib.sha256(text.encode('utf-8')).hexdigest(), 'annotation_config': annotation_config, 'spans': [asdict(s) for s in spans]}

def cached_segment(text, cache, segmenter=segment, annotation_config=None):
    if cache and cache.get('text_hash') == hashlib.sha256(text.encode('utf-8')).hexdigest() and (annotation_config is None or cache.get('annotation_config') == annotation_config):
        try:
            if annotation_config is not None and any(not isinstance(s, dict) or not {'id','start','end','kind','scope','modifies_id','preconditions'} <= s.keys() or not isinstance(s['preconditions'], (list, tuple)) for s in cache['spans']):
                raise ValueError('incomplete annotation cache')
            spans = [Span(**{**s, 'preconditions': tuple(s.get('preconditions', ()))}) for s in cache['spans']]
            if validate(text, spans) and (annotation_config is None or valid_annotations(spans, annotation_config.get('preconditions', False))):
                return spans
        except (ValueError, TypeError, KeyError, AttributeError):
            pass
    spans = segmenter(text)
    if not validate(text, spans):
        raise ValueError('invalid segmentation')
    return spans


def valid_annotations(spans, preconditions):
    ids = {s.id for s in spans}
    return all(s.kind in {'ASSERTION', 'CONDITION'} and s.scope in {'NONE', 'LOCAL', 'FORWARD', 'GLOBAL'}
               and (s.modifies_id is None or (s.modifies_id in ids and s.modifies_id != s.id))
               and isinstance(s.preconditions, tuple) and all(isinstance(p, str) and p.strip() for p in s.preconditions)
               and (not preconditions or s.kind != 'ASSERTION' or 1 <= len(s.preconditions) <= 2)
               and (preconditions or not s.preconditions) for s in spans)
