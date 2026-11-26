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
        if s.id in ids or s.start != pos or not s.start < s.end <= len(text):
            return False
        ids.add(s.id)
        pos = s.end
    return pos == len(text) and ''.join(text[s.start:s.end] for s in spans) == text

def cache_record(text, spans):
    if not validate(text, spans):
        raise ValueError('invalid spans')
    return {'text_hash': hashlib.sha256(text.encode('utf-8')).hexdigest(), 'spans': [asdict(s) for s in spans]}

def cached_segment(text, cache, segmenter=segment):
    if cache and cache.get('text_hash') == hashlib.sha256(text.encode('utf-8')).hexdigest():
        try:
            spans = [Span(**{**s, 'preconditions': tuple(s.get('preconditions', ()))}) for s in cache['spans']]
            if validate(text, spans):
                return spans
        except (ValueError, TypeError, KeyError):
            pass
    spans = segmenter(text)
    if not validate(text, spans):
        raise ValueError('invalid segmentation')
    return spans
