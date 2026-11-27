from types import SimpleNamespace
import pytest
from casp_mem.schemas import PatchOp, Span, ScopeDecision
from casp_mem.patching.patch_generator import generate
from casp_mem.segmentation.segmenter import cache_record, cached_segment
from casp_mem.integration.mem0_adapter import Mem0Adapter


def test_empty_prune_only_with_explicit_mode_and_grounded_decision():
    llm = SimpleNamespace(patches=lambda *a, **k: [PatchOp(0, 'PRUNE_OR_ADAPT', '', op_mode='PRUNE')])
    decisions = [ScopeDecision(0, 'CASCADED_INVALID', 'evidence invalidates this clause')]
    assert generate(llm, None, 'bad.', 'new', [Span(0, 0, 4)], decisions, {'patch_check': True})[0].text == ''
    with pytest.raises(ValueError):
        generate(SimpleNamespace(patches=lambda *a, **k: [PatchOp(0, 'PRUNE_OR_ADAPT', '')]), None, 'bad.', 'new', [Span(0, 0, 4)], decisions, {})


def test_annotation_cache_checks_configuration_and_contents():
    spans = [Span(0, 0, 4, 'ASSERTION', 'LOCAL', None, ('premise',))]
    cache = cache_record('text', spans, annotation_config={'model':'x', 'preconditions':True})
    assert cached_segment('text', cache, lambda _: pytest.fail('cache missed'), annotation_config={'model':'x', 'preconditions':True}) == spans
    calls = []
    cached_segment('text', cache, lambda text: calls.append(text) or [Span(0,0,4)], annotation_config={'model':'y', 'preconditions':True})
    assert calls == ['text']
    cache['spans'][0]['modifies_id'] = 99
    assert cached_segment('text', cache, lambda _: [Span(0,0,4)], annotation_config={'model':'x', 'preconditions':True}) == [Span(0,0,4)]


def test_comparable_scores_take_max_and_filter_foreign_stale_parents():
    payloads = {'a': {'data':'a', 'user_id':'u', 'casp_version':1}, 'b': {'data':'b','user_id':'u','casp_version':42}, 'c': {'data':'c','user_id':'other'}}
    store = SimpleNamespace(get=lambda vector_id: SimpleNamespace(payload=payloads[vector_id]) if vector_id in payloads else None)
    memory = SimpleNamespace(vector_store=store, search=lambda *a, **k: {'results':[{'id':'a','memory':'a','score':.2}, {'id':'b','memory':'b','score':.8}]})
    index = SimpleNamespace(search=lambda *a: [{'parent_id':'a','version':1,'user_id':'u','score':.9}, {'parent_id':'b','version':41,'user_id':'u','score':1.0}, {'parent_id':'c','version':0,'user_id':'u','score':1.0}, {'parent_id':'gone','version':0,'user_id':'u','score':1.0}])
    result = Mem0Adapter(memory, None, index, comparable_scores=True).search('x', user_id='u')
    assert [(x['id'],x['score']) for x in result] == [('a',.9), ('b',.8)]


def test_inferred_delete_uses_guard_with_raw_evidence_and_preserves_uncertain():
    from casp_mem.integration.delete_guard import DeleteGuard
    point = SimpleNamespace(payload={'data':'tea and music','user_id':'u'})
    memory = SimpleNamespace(vector_store=SimpleNamespace(get=lambda **kw: point))
    seen = []
    judge = SimpleNamespace(classify_delete=lambda old, evidence: seen.append((old,evidence)) or {'decision':'UNCERTAIN', 'reason':'ambiguous'})
    adapter = Mem0Adapter(memory, SimpleNamespace(llm=judge), multi_anchor=False)
    result = DeleteGuard(adapter).handle('id', 'raw evidence', 'u')
    assert result['event'] == 'RETAIN' and seen == [('tea and music','raw evidence')]


def test_partial_delete_runs_pipeline_and_preserves_locked_characters():
    from casp_mem.pipeline import Pipeline
    class LLM:
        def classify_delete(self, old, evidence):
            return {'decision':'PRUNE', 'reason':'first fact invalid', 'invalid_span_ids':[0]}
        def annotate(self, text, spans, enabled):
            return [{'id':s.id,'kind':'ASSERTION','scope':'NONE','modifies_id':None,'preconditions':[]} for s in spans]
        def review(self, text, evidence, spans, selected):
            return [{'span_id':s.id,'label':'CASCADED_INVALID' if s.id in selected else 'LOCKED_INVARIANT','reason':'invalid fact' if s.id in selected else 'still valid'} for s in spans]
        def patches(self, evidence, context, decisions, split, repair=None):
            return [PatchOp(d.span_id,'PRUNE_OR_ADAPT','',op_mode='PRUNE') for d in decisions if d.label != 'LOCKED_INVARIANT']
    class Memory:
        def __init__(self): self.payload={'data':'bad.  Good!', 'user_id':'u'}; self.vector_store=self
        def get(self, vector_id): return SimpleNamespace(payload=self.payload.copy())
        def _update_memory(self, memory_id, data, existing_embeddings, metadata): self.payload={**metadata, 'data':data}
    nli = SimpleNamespace(score=lambda p,h: {'contradiction':0.,'entailment':1.})
    memory = Memory()
    from casp_mem.integration.delete_guard import DeleteGuard
    adapter = Mem0Adapter(memory, Pipeline(LLM(), nli, {'preconditions':False, 'counterfactual':False, 'hard_lock':False}), multi_anchor=False)
    output = DeleteGuard(adapter).handle('id', 'forget the first fact', 'u')
    assert output['event'] == 'UPDATE' and memory.payload['data'] == '  Good!'


def test_full_delete_cleans_anchors_and_partial_write_errors_propagate():
    from casp_mem.integration.delete_guard import DeleteGuard
    from casp_mem.integration.mem0_adapter import PartialDeleteError
    class Memory:
        def __init__(self): self.payload={'data':'first. second.', 'user_id':'u'}; self.vector_store=self
        def get(self, vector_id): return SimpleNamespace(payload=self.payload) if self.payload else None
        def _delete_memory(self, memory_id): self.payload=None
    class Index:
        def __init__(self): self.calls=[]
        def replace(self, *args): self.calls.append(args)
    memory, index = Memory(), Index()
    llm = SimpleNamespace(classify_delete=lambda *a: {'decision':'FULL','whole_confirmed':True,'reason':'explicit full request'})
    adapter = Mem0Adapter(memory, SimpleNamespace(llm=llm), index)
    assert DeleteGuard(adapter).handle('id', 'forget this whole memory', 'u')['event'] == 'DELETE'
    assert memory.payload is None and index.calls == [('id','u',0,[])]
    memory = Memory()
    def failed_delete(_):
        memory.payload = None
        raise RuntimeError('history failed')
    memory._delete_memory = failed_delete
    with pytest.raises(PartialDeleteError, match='parent absent, anchors removed'):
        Mem0Adapter(memory, SimpleNamespace(llm=llm), Index()).delete('id', user_id='u')


def test_inferred_delete_from_add_receives_original_evidence(tmp_path):
    from scripts.run_casp_update import import_mem0
    import_mem0({'paths': {'mem0':'mem0_v0.1.118_local'}}, tmp_path / 'mem0')
    class LLM:
        def __init__(self): self.calls=0
        def generate_response(self, **kw):
            self.calls += 1
            return '{"facts":["extracted"]}' if self.calls == 1 else '{"memory":[{"id":"0","event":"DELETE","text":"old"}]}'
    memory = SimpleNamespace(llm=LLM(), config=SimpleNamespace(custom_fact_extraction_prompt=None, custom_update_memory_prompt=None), embedding_model=SimpleNamespace(embed=lambda *a:[0]), search=lambda *a,**k:{'results':[{'id':'id','memory':'old','score':1}]}, vector_store=SimpleNamespace(get=lambda **k:SimpleNamespace(payload={'data':'old','user_id':'u'})))
    seen = []
    judge = SimpleNamespace(classify_delete=lambda old, evidence: seen.append(evidence) or {'decision':'UNCERTAIN','reason':'not enough evidence'})
    result = Mem0Adapter(memory, SimpleNamespace(llm=judge), multi_anchor=False).add('original user evidence', user_id='u')
    assert result['results'][0]['event'] == 'RETAIN'
    assert seen == ['original user evidence']


def test_pipeline_reuses_complete_cache_and_invalidates_changed_annotation_config():
    from casp_mem.pipeline import Pipeline
    class LLM:
        annotation_cache_key = {'model':'test', 'prompt_version':1}
        def __init__(self): self.calls=[]
        def annotate(self, text, spans, enabled):
            self.calls.append((text, enabled))
            return [{'id':s.id,'kind':'ASSERTION','scope':'LOCAL','modifies_id':None,'preconditions':['premise'] if enabled else []} for s in spans]
        def review(self, text, evidence, spans, selected):
            return [{'span_id':s.id,'label':'ATTR_CONFLICT' if s.id in selected else 'LOCKED_INVARIANT','reason':'conflict'} for s in spans]
        def patches(self, evidence, context, decisions, split, repair=None):
            return [PatchOp(d.span_id,'REPLACE_ATTR','new.') for d in decisions if d.label != 'LOCKED_INVARIANT']
    # The replacement must also pass the final audit.
    class SafeNLI:
        def score(self, p, h): return {'contradiction':1. if p == 'evidence' and h == 'old.' else 0., 'entailment':1.}
    llm = LLM()
    pipeline = Pipeline(llm, SafeNLI(), {'counterfactual':False, 'patch_check':False})
    first = pipeline.run('old.', 'evidence')
    assert first.cache['spans'][0]['preconditions'] == ('premise',)
    llm.calls.clear()
    # No conflict is needed to prove reuse: inspect annotation calls after rejection.
    from casp_mem.pipeline import Rejected
    with pytest.raises(Rejected): pipeline.run('new.', 'evidence', first.cache)
    assert llm.calls == []
    pipeline.options['preconditions'] = False
    with pytest.raises(Rejected): pipeline.run('new.', 'evidence', first.cache)
    assert llm.calls == [('new.', False)]


def test_pruned_candidate_is_audited_after_joining_locked_text():
    from casp_mem.guard.fidelity_guard import audit
    from casp_mem.patching.assembler import assemble
    text = 'bad. Keep!'
    spans = [Span(0,0,4), Span(1,4,len(text))]
    candidate, locations = assemble(text, spans, [PatchOp(0,'PRUNE_OR_ADAPT','',op_mode='PRUNE')], {1})
    assert candidate == ' Keep!'
    nli = SimpleNamespace(score=lambda p,h: {'contradiction':1. if p == candidate and h == 'evidence' else 0., 'entailment':1.})
    with pytest.raises(ValueError, match='candidate conflicts'):
        audit(nli, text, candidate, 'evidence', spans, {1}, locations)


def test_unknown_score_space_keeps_main_first_without_comparing_scores():
    payloads = {'old':{'data':'old','user_id':'u'}, 'new':{'data':'new','user_id':'u','casp_version':1,'casp_score_space':'v2'}}
    memory = SimpleNamespace(vector_store=SimpleNamespace(get=lambda vector_id:SimpleNamespace(payload=payloads[vector_id])), search=lambda *a,**k:{'results':[{'id':'old','memory':'old','score':.1}]})
    hits = [{'parent_id':'new','version':1,'user_id':'u','score':.99}, {'parent_id':'new','version':1,'user_id':'other','score':1.0}]
    adapter = Mem0Adapter(memory, None, SimpleNamespace(search=lambda *a:hits), comparable_scores=True, score_space='v2')
    assert [(item['id'], item['score']) for item in adapter.search('query', user_id='u')] == [('old',.1), ('new',.99)]


def test_partial_update_after_parent_write_is_not_reported_as_retain():
    from casp_mem.integration.delete_guard import DeleteGuard
    from casp_mem.integration.mem0_adapter import PartialWriteError
    class Memory:
        def __init__(self): self.payload={'data':'bad. good.','user_id':'u'}; self.vector_store=self
        def get(self, vector_id): return SimpleNamespace(payload=self.payload.copy())
        def _update_memory(self, memory_id, data, existing_embeddings, metadata):
            self.payload={**metadata,'data':data}
            raise ValueError('history failed')
    class Pipeline:
        llm = SimpleNamespace(classify_delete=lambda *a:{'decision':'PRUNE','reason':'first invalid','invalid_span_ids':[0]})
        def run(self, *args, **kwargs): return SimpleNamespace(text=' good.', cache=None, anchors=[], patches=[])
    memory = Memory()
    with pytest.raises(PartialWriteError, match='was written'):
        DeleteGuard(Mem0Adapter(memory, Pipeline(), multi_anchor=False)).handle('id', 'new evidence', 'u')
    assert memory.payload['data'] == ' good.'
