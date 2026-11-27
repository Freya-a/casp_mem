from types import SimpleNamespace
import pytest
from casp_mem.integration.mem0_adapter import Mem0Adapter, PartialSyncError
from casp_mem.pipeline import Rejected

class Memory:
    def __init__(self):
        self.payload = {'data':'old', 'user_id':'u', 'custom':'retained'}
        self.vector_store = self
        self.writes = []
    def get(self, vector_id): return SimpleNamespace(payload=self.payload.copy())
    def _update_memory(self, memory_id, data, existing_embeddings, metadata):
        self.writes.append(data); self.payload = {**metadata, 'data':data}

class Pipeline:
    def __init__(self, fail=False): self.fail=fail
    def run(self, old, evidence, cache):
        if self.fail: raise Rejected('audit failed')
        assert evidence == 'raw input'
        return SimpleNamespace(text='actual candidate', anchors=['Tuesday'], cache={'v':1}, patches=[])

class Index:
    def __init__(self, fail=False): self.fail=fail; self.calls=[]
    def search(self, query, user_id, limit): return []
    def replace(self, memory_id, scope, version, anchors):
        self.calls.append((memory_id, scope, version, anchors))
        if self.fail: raise RuntimeError('index down')


def test_write_result_and_preserve_metadata():
    memory, index = Memory(), Index()
    result = Mem0Adapter(memory, Pipeline(), index).update('id', raw_evidence='raw input', user_id='u')
    assert result['memory'] == 'actual candidate'
    assert memory.payload['custom'] == 'retained'
    assert memory.payload['casp_version'] == 1
    assert index.calls == [('id','u',1,['Tuesday'])]


def test_reject_without_write_and_propagate():
    memory = Memory()
    with pytest.raises(Rejected): Mem0Adapter(memory, Pipeline(True), Index()).update('id', raw_evidence='raw input', user_id='u')
    assert memory.writes == []
    with pytest.raises(ValueError): Mem0Adapter(memory, Pipeline(), Index()).update('id', raw_evidence='', user_id='u')


def test_partial_sync_can_be_recovered():
    memory, index = Memory(), Index(True)
    adapter = Mem0Adapter(memory, Pipeline(), index)
    with pytest.raises(PartialSyncError): adapter.update('id', raw_evidence='raw input', user_id='u')
    assert memory.writes == ['actual candidate']
    index.fail = False
    adapter.recover('id', user_id='u')
    assert index.calls[-1] == ('id','u',1,['Tuesday'])


def test_anchor_search_filters_stale_and_deduplicates():
    memory = Memory()
    memory.payload.update(casp_version=3)
    memory.search = lambda *a,**k: {'results':[{'id':'id','memory':'old','score':0.4}]}
    index = Index()
    index.search = lambda *a: [{'parent_id':'id','version':3,'user_id':'u','score':.9}, {'parent_id':'id','version':2,'user_id':'u','score':.99}]
    result = Mem0Adapter(memory, Pipeline(), index).search('query', user_id='u')
    assert len(result) == 1 and result[0]['id'] == 'id'
    memory.search = lambda *a,**k: {'results':[]}
    index.search = lambda *a: [{'parent_id':'id','version':2,'user_id':'u','score':.99}]
    assert Mem0Adapter(memory, Pipeline(), index).search('query', user_id='u') == []


def test_multi_anchor_ablation_never_calls_index():
    memory, index = Memory(), Index(True)
    result = Mem0Adapter(memory, Pipeline(), index, multi_anchor=False).update('id', raw_evidence='raw input', user_id='u')
    assert result['memory'] == 'actual candidate' and not index.calls


def test_result_exposes_patches_and_anchors():
    from casp_mem.schemas import PatchOp
    pipeline = Pipeline()
    pipeline.run = lambda *args: SimpleNamespace(text='actual candidate', anchors=['Tuesday'], cache={'v':1}, patches=[PatchOp(0,'SPLIT_CONDITION','Tuesday',('Tuesday',))])
    result = Mem0Adapter(Memory(), pipeline, Index()).update('id', raw_evidence='raw input', user_id='u')
    assert result['patches'][0]['operation'] == 'SPLIT_CONDITION'
    assert result['anchors'] == ['Tuesday']


def test_mem0_action_update_uses_raw_evidence_and_returns_written_text(tmp_path):
    from scripts.run_casp_update import import_mem0
    cfg = {'paths': {'mem0': 'mem0_v0.1.118_local'}}
    import_mem0(cfg, tmp_path / 'mem0')
    class LLM:
        def __init__(self): self.calls = 0
        def generate_response(self, **kwargs):
            self.calls += 1
            if self.calls == 1: return '{"facts":["extracted"]}'
            return '{"memory":[{"id":"0","event":"UPDATE","text":"LLM whole rewrite"}]}'
    memory = Memory()
    memory.llm = LLM()
    memory.config = SimpleNamespace(custom_fact_extraction_prompt=None, custom_update_memory_prompt=None)
    memory.embedding_model = SimpleNamespace(embed=lambda *args: [0.1])
    memory.search = lambda *a,**kw: {'results':[{'id':'id', 'memory':'old', 'score':1.0}]}
    adapter = Mem0Adapter(memory, Pipeline(), Index())
    result = adapter.add('raw input', user_id='u')
    assert result['results'][0]['memory'] == 'actual candidate'
    assert memory.writes == ['actual candidate']


def test_mem0_action_propagates_rejection(tmp_path):
    from scripts.run_casp_update import import_mem0
    import_mem0({'paths': {'mem0': 'mem0_v0.1.118_local'}}, tmp_path / 'mem0')
    class LLM:
        def __init__(self): self.calls=0
        def generate_response(self, **kw):
            self.calls += 1
            return '{"facts":["x"]}' if self.calls == 1 else '{"memory":[{"id":"0","event":"UPDATE","text":"rewrite"}]}'
    memory = Memory(); memory.llm=LLM()
    memory.config=SimpleNamespace(custom_fact_extraction_prompt=None, custom_update_memory_prompt=None)
    memory.embedding_model=SimpleNamespace(embed=lambda *args: [0])
    memory.search=lambda *args,**kwargs: {'results':[{'id':'id','memory':'old','score':1}]}
    with pytest.raises(Rejected, match='audit failed'):
        Mem0Adapter(memory, Pipeline(True), Index()).add('raw input', user_id='u')
    assert memory.writes == []


def test_batch_failure_reports_already_completed_actions(tmp_path):
    from casp_mem.integration.mem0_adapter import PartialBatchError
    from scripts.run_casp_update import import_mem0
    import_mem0({'paths': {'mem0': 'mem0_v0.1.118_local'}}, tmp_path / 'mem0')
    class LLM:
        def __init__(self): self.calls=0
        def generate_response(self, **kw):
            self.calls += 1
            return '{"facts":["x"]}' if self.calls == 1 else '{"memory":[{"event":"ADD","text":"first"},{"id":"0","event":"UPDATE","text":"rewrite"}]}'
    memory = Memory(); memory.llm=LLM()
    memory.config=SimpleNamespace(custom_fact_extraction_prompt=None, custom_update_memory_prompt=None)
    memory.embedding_model=SimpleNamespace(embed=lambda *args: [0])
    memory.search=lambda *args,**kwargs: {'results':[{'id':'id','memory':'old','score':1}]}
    memory._create_memory=lambda *args: 'new-id'
    with pytest.raises(PartialBatchError) as error:
        Mem0Adapter(memory, Pipeline(True), Index()).add('raw input', user_id='u')
    assert error.value.completed[0]['id'] == 'new-id'


def test_copied_mem0_writes_sqlite_history_and_preserves_metadata(tmp_path, monkeypatch):
    from scripts.run_casp_update import import_mem0
    mem0 = import_mem0({'paths': {'mem0': 'mem0_v0.1.118_local'}}, tmp_path / 'internal')
    import mem0.memory.main as main
    from mem0.memory.storage import SQLiteManager
    monkeypatch.setattr(main, 'capture_event', lambda *args: None)
    class Store:
        def __init__(self): self.payload = {'data':'old','user_id':'u','custom':'retained','created_at':'yesterday'}
        def get(self, vector_id): return SimpleNamespace(payload=self.payload.copy())
        def update(self, vector_id, vector, payload): self.payload = payload
    real = mem0.Memory.__new__(mem0.Memory)
    real.vector_store=Store()
    real.embedding_model=SimpleNamespace(embed=lambda *args: [1.0])
    real.db=SQLiteManager(str(tmp_path / 'history.sqlite'))
    real.db.add_history('id', None, 'old', 'ADD')
    output = Mem0Adapter(real, Pipeline(), Index()).update('id', raw_evidence='raw input', user_id='u')
    rows = real.db.connection.execute('SELECT old_memory,new_memory,event FROM history ORDER BY rowid').fetchall()
    assert rows == [(None,'old','ADD'),('old','actual candidate','UPDATE')]
    assert real.vector_store.payload['custom'] == 'retained'
    assert output['memory'] == real.vector_store.payload['data']
    real.db.connection.close()


def test_delete_failure_reports_anchor_state_and_recovery():
    from casp_mem.integration.mem0_adapter import PartialDeleteError
    memory, index = Memory(), Index()
    memory._delete_memory = lambda _: (_ for _ in ()).throw(RuntimeError('delete failed'))
    adapter = Mem0Adapter(memory, Pipeline(), index)
    with pytest.raises(PartialDeleteError, match='recover'):
        adapter.delete('id', user_id='u')
    assert index.calls == [('id','u',0,[])]
