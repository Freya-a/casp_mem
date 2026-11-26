import pytest
from casp_mem.pipeline import Pipeline, Rejected
from casp_mem.schemas import PatchOp

class FakeLLM:
    def __init__(self, label='ATTR_CONFLICT', operation='REPLACE_ATTR', text='Likes coffee.', anchors=()):
        self.label, self.operation, self.text, self.anchors = label, operation, text, anchors
        self.preconditions_requested = None
    def annotate(self, text, spans, preconditions):
        self.preconditions_requested = preconditions
        return [{'id': s.id, 'kind':'ASSERTION', 'scope':'NONE', 'modifies_id':None, 'preconditions':['Likes tea'] if preconditions else []} for s in spans]
    def review(self, text, evidence, spans, selected):
        return [{'span_id':s.id,'label':self.label if s.id in selected else 'LOCKED_INVARIANT','reason':'test'} for s in spans]
    def patches(self, evidence, context, decisions, split, repair=None):
        return [PatchOp(d.span_id, self.operation, self.text, self.anchors) for d in decisions if d.label != 'LOCKED_INVARIANT']

class FakeNLI:
    def __init__(self): self.calls=[]
    def score(self, premise, hypothesis):
        self.calls.append((premise,hypothesis))
        return {'contradiction': 0.95 if premise == 'Likes coffee.' and ('Likes tea' in hypothesis) else 0.0, 'entailment': 0.8}


def test_pipeline_replaces_and_preconditions_switch():
    llm, nli = FakeLLM(), FakeNLI()
    result = Pipeline(llm, nli, {'preconditions': False, 'counterfactual': False, 'patch_check': False}).run('Likes tea.', 'Likes coffee.')
    assert result.text == 'Likes coffee.'
    assert llm.preconditions_requested is False
    assert not any('Likes tea' == h for p,h in nli.calls if p != 'Likes coffee.')


def test_pipeline_refuses_uncertain_and_invalid_split():
    with pytest.raises(Rejected):
        Pipeline(FakeLLM(label='UNCERTAIN'), FakeNLI(), {}).run('Likes tea.', 'Likes coffee.')
    with pytest.raises(Rejected):
        Pipeline(FakeLLM(operation='SPLIT_CONDITION', anchors=('invented',)), FakeNLI(), {}).run('Likes tea.', 'Likes coffee.')
    with pytest.raises(Rejected):
        Pipeline(FakeLLM(operation='SPLIT_CONDITION', anchors=('Likes',)), FakeNLI(), {'split_condition':False}).run('Likes tea.', 'Likes coffee.')


def test_patch_check_and_final_audit_reject():
    class NLI(FakeNLI):
        def score(self, p, h):
            if p == 'bad patch': return {'contradiction': 1.0, 'entailment': 0.0}
            return super().score(p, h)
    with pytest.raises(Rejected):
        Pipeline(FakeLLM(text='bad patch'), NLI(), {}).run('Likes tea.', 'Likes coffee.')


def test_condition_split_requires_explicit_condition():
    class SplitLLM(FakeLLM):
        def review(self, text, evidence, spans, selected):
            return [{'span_id':s.id,'label':'CONDITION_SHIFT' if s.id in selected else 'LOCKED_INVARIANT','reason':'explicit schedule'} for s in spans]
        def patches(self, evidence, context, decisions, split, repair=None):
            return [PatchOp(d.span_id, 'SPLIT_CONDITION', 'Likes tea normally, on Tuesday Likes coffee.', ('on Tuesday',), 'Likes tea', 'Likes coffee') for d in decisions if d.label != 'LOCKED_INVARIANT']
    class ConditionalNLI(FakeNLI):
        def score(self, p, h):
            if p == 'Likes coffee. on Tuesday' and h in ('Likes tea.', 'Likes tea'):
                return {'contradiction': 0.95, 'entailment': 0.0}
            return {'contradiction': 0.0, 'entailment': 0.8}
    nli = ConditionalNLI()
    llm = SplitLLM(operation='SPLIT_CONDITION', text='Likes tea normally, on Tuesday likes coffee.', anchors=('on Tuesday',))
    result = Pipeline(llm, nli, {'patch_check':False}).run('Likes tea.', 'Likes coffee. on Tuesday')
    assert result.anchors == ['on Tuesday']
    with pytest.raises(Rejected): Pipeline(llm, nli, {'patch_check':False, 'split_condition':False}).run('Likes tea.', 'Likes coffee. on Tuesday')


def test_condition_shift_cannot_be_disguised_as_replace():
    llm = FakeLLM(label='CONDITION_SHIFT')
    with pytest.raises(Rejected): Pipeline(llm, FakeNLI(), {'patch_check':False, 'split_condition':False}).run('Likes tea.', 'Likes coffee.')


def test_precondition_conflict_can_be_seed():
    class NLI(FakeNLI):
        def score(self, p, h):
            self.calls.append((p,h))
            return {'contradiction': 0.95 if p == 'raw' and h == 'Likes tea' else 0.0, 'entailment': 0.8}
    llm = FakeLLM(text='Updated.')
    assert Pipeline(llm, NLI(), {'patch_check':False}).run('This is an assertion.', 'raw').text == 'Updated.'
    with pytest.raises(Rejected): Pipeline(llm, NLI(), {'patch_check':False, 'preconditions':False}).run('This is an assertion.', 'raw')


def test_patch_check_ablation_does_not_disable_final_audit():
    class NLI(FakeNLI):
        def score(self, p, h):
            if p == 'bad patch' and h == 'Likes coffee.': return {'contradiction': 1.0, 'entailment': 0.0}
            return super().score(p,h)
    with pytest.raises(Rejected): Pipeline(FakeLLM(text='bad patch'), NLI(), {'patch_check':False}).run('Likes tea.', 'Likes coffee.')


def test_counterfactual_excludes_target_span_and_switch():
    from casp_mem.grounding.counterfactual import sensitivities
    from casp_mem.schemas import Span
    nli = FakeNLI()
    old = 'seed|dependent'
    spans = [Span(0,0,5), Span(1,5,len(old))]
    sensitivities(nli, old, 'new', spans, {0})
    assert nli.calls == [('seed|','dependent'), ('new','dependent')]


def test_hard_lock_ablation_changes_merge_whitespace():
    llm = FakeLLM(text=' New  value. ')
    assert Pipeline(llm, FakeNLI(), {'hard_lock':True, 'patch_check':False}).run('Likes tea.', 'Likes coffee.').text == ' New  value. '
    assert Pipeline(llm, FakeNLI(), {'hard_lock':False, 'patch_check':False}).run('Likes tea.', 'Likes coffee.').text == 'New value.'


def test_counterfactual_switch_controls_computation(monkeypatch):
    import casp_mem.pipeline as module
    calls = []
    def spy(nli, old, evidence, spans, seeds):
        calls.append(True)
        return {s.id: 0.0 for s in spans}
    monkeypatch.setattr(module, 'sensitivities', spy)
    Pipeline(FakeLLM(), FakeNLI(), {'counterfactual':False, 'patch_check':False}).run('Likes tea.', 'Likes coffee.')
    assert calls == []
    Pipeline(FakeLLM(), FakeNLI(), {'counterfactual':True, 'patch_check':False}).run('Likes tea.', 'Likes coffee.')
    assert calls == [True]


def test_disabling_patch_check_removes_patch_only_scoring():
    class NLI(FakeNLI):
        def score(self,p,h):
            self.calls.append((p,h))
            return {'contradiction': 0.95 if p == 'Likes coffee.' and h == 'patch' else (0.95 if p == 'Likes coffee.' and h == 'Likes tea.' else 0), 'entailment': .8}
    # Candidate is padded to make whole-text audit different from patch-level input.
    llm = FakeLLM(text='patch')
    nli = NLI()
    with pytest.raises(Rejected): Pipeline(llm, nli, {}).run('Likes tea. Extra.', 'Likes coffee.')
    assert ('Likes coffee.', 'patch') in nli.calls
    nli2 = NLI()
    Pipeline(llm, nli2, {'patch_check':False}).run('Likes tea. Extra.', 'Likes coffee.')
    assert ('Likes coffee.', 'patch') not in nli2.calls


def test_condition_trigger_can_seed_without_contradiction():
    from casp_mem.grounding.seed_detector import detect
    from casp_mem.schemas import Span
    class NLI:
        def score(self, p, h): return {'contradiction':0, 'entailment':0.95 if p == 'It is Tuesday' and h == 'on Tuesday' else 0}
    spans = [Span(0,0,10,kind='CONDITION')]
    assert detect(NLI(), 'It is Tuesday', 'on Tuesday', spans, .8)[0] == {0}


def test_audit_rejects_low_local_entailment():
    from casp_mem.guard.fidelity_guard import audit
    from casp_mem.schemas import Span
    class NLI:
        def score(self,p,h): return {'contradiction':0, 'entailment':0.1}
    with pytest.raises(ValueError, match='local window'):
        audit(NLI(), 'old', 'old', 'new', [Span(0,0,3)], {0}, {0:(0,3)})


def test_condition_split_rejects_missing_old_and_new_branches():
    from casp_mem.patching.patch_generator import generate
    from casp_mem.schemas import Span, ScopeDecision
    class BadLLM:
        def patches(self,*args,**kwargs):
            return [PatchOp(0,'SPLIT_CONDITION','on Tuesday coffee.',('on Tuesday',))]
    with pytest.raises(ValueError, match='branch'):
        generate(BadLLM(), FakeNLI(), 'Likes tea.', 'on Tuesday likes coffee', [Span(0,0,10)], [ScopeDecision(0,'CONDITION_SHIFT','new schedule')], {'patch_check':False})


def test_invalid_annotation_refuses_instead_of_inventing_preconditions():
    llm = FakeLLM()
    llm.annotate = lambda *args: [{'id':0,'kind':'ASSERTION','scope':'NONE','modifies_id':None,'preconditions':[]}]
    with pytest.raises(Rejected, match='preconditions'):
        Pipeline(llm, FakeNLI(), {}).run('Likes tea.', 'Likes coffee.')
