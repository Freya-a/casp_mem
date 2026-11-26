import pytest
from casp_mem.segmentation.segmenter import segment, validate, cached_segment
from casp_mem.grounding.scope_solver import solve
from casp_mem.patching.assembler import assemble
from casp_mem.schemas import Span, PatchOp


def test_lossless_unicode_repetition_and_cache():
    text = '如果下雨，喝茶。 如果下雨，喝茶！🍵'
    spans = segment(text)
    assert ''.join(text[s.start:s.end] for s in spans) == text
    assert validate(text, spans)
    calls = []
    assert cached_segment(text, {'text_hash': 'wrong', 'spans': []}, lambda t: calls.append(t) or spans) == spans
    assert calls == [text]
    assert cached_segment(text, {'text_hash': __import__('hashlib').sha256(text.encode()).hexdigest(), 'spans': [vars(s) for s in spans]}, lambda _: pytest.fail('cache missed')) == spans


def test_seed_forced_and_dependency_propagates():
    assert solve([10, 0], [0, 0.1], [(0, 1, 5)], {0}, 1) == {0, 1}
    with pytest.raises(ValueError):
        solve([1], [1], [], set(), 1)


def test_hard_lock_exact_and_invalid_patch_rejected():
    text = 'Alice likes tea. Bob likes coffee.'
    spans = [Span(0, 0, 16), Span(1, 16, len(text))]
    out, locations = assemble(text, spans, [PatchOp(0, 'REPLACE_ATTR', 'Alice likes juice.')], {1})
    assert out[locations[1][0]:locations[1][1]] == text[16:]
    with pytest.raises(ValueError):
        assemble(text, spans, [PatchOp(1, 'REPLACE_ATTR', 'changed')], {1})
