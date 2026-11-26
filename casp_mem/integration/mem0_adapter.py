import json
import threading
from dataclasses import asdict
from copy import deepcopy

class PartialSyncError(RuntimeError):
    def __init__(self, memory_id, version, reason):
        self.memory_id, self.version = memory_id, version
        super().__init__(f'main memory {memory_id} version {version} was written; anchor synchronization failed: {reason}; run recover')

class PartialWriteError(RuntimeError):
    pass

class PartialDeleteError(RuntimeError):
    pass

class PartialBatchError(RuntimeError):
    def __init__(self, completed, cause):
        self.completed = completed
        self.cause = cause
        super().__init__(f'{len(completed)} prior action(s) completed; subsequent action failed: {cause}')

class Mem0Adapter:
    """Use this adapter instead of Memory.add/update. Evidence is passed as a call argument."""
    def __init__(self, memory, pipeline, anchors=None, multi_anchor=True):
        self.memory, self.pipeline, self.anchors = memory, pipeline, anchors
        self.multi_anchor = multi_anchor
        self._lock = threading.RLock()
        if multi_anchor and anchors is None:
            raise ValueError('multi-anchor mode requires an independent index')

    def _get(self, memory_id, user_id):
        point = self.memory.vector_store.get(vector_id=memory_id)
        if not point or point.payload.get('user_id') != user_id:
            raise ValueError('memory not found in user scope')
        return point

    def update(self, memory_id, *, raw_evidence, user_id):
        if not isinstance(raw_evidence, str) or not raw_evidence.strip() or not user_id:
            raise ValueError('explicit raw evidence and user_id required')
        with self._lock:
            point = self._get(memory_id, user_id)
            payload = deepcopy(point.payload)
            result = self.pipeline.run(payload['data'], raw_evidence, payload.get('casp_span_cache'))
            version = int(payload.get('casp_version', 0)) + 1
            payload.update(casp_version=version, casp_span_cache=result.cache, casp_anchors=result.anchors if self.multi_anchor else [])
            try:
                self.memory._update_memory(memory_id, result.text, {}, metadata=payload)
            except Exception as e:
                current = self.memory.vector_store.get(vector_id=memory_id)
                if current and current.payload.get('casp_version') == version and current.payload.get('data') == result.text:
                    raise PartialWriteError(f'main memory {memory_id} version {version} was written but Mem0 history/write failed; inspect history and run recover') from e
                raise
            if self.multi_anchor:
                try:
                    self.anchors.replace(memory_id, user_id, version, result.anchors)
                except Exception as e:
                    raise PartialSyncError(memory_id, version, str(e)) from e
            return {'id':memory_id, 'event':'UPDATE', 'memory':result.text, 'version':version, 'patches':[asdict(p) for p in result.patches], 'anchors':result.anchors, 'anchors_synchronized':self.multi_anchor}

    def recover(self, memory_id, *, user_id):
        with self._lock:
            payload = self._get(memory_id, user_id).payload
            if not self.multi_anchor:
                raise ValueError('multi-anchor mode disabled')
            self.anchors.replace(memory_id, user_id, payload.get('casp_version', 0), payload.get('casp_anchors', []))
            return {'id': memory_id, 'version':payload.get('casp_version', 0), 'anchors_synchronized':True}

    def delete(self, memory_id, *, user_id):
        with self._lock:
            self._get(memory_id, user_id)
            try:
                if self.multi_anchor:
                    self.anchors.replace(memory_id, user_id, 0, [])
                self.memory._delete_memory(memory_id)
            except Exception as exc:
                raise PartialDeleteError(f'delete of {memory_id} may be partial; if the parent exists run recover, otherwise inspect SQLite history and anchor collection: {exc}') from exc
            return {'id':memory_id, 'event':'DELETE'}

    def search(self, query, *, user_id, limit=10):
        if not user_id:
            raise ValueError('user_id required')
        main = self.memory.search(query, user_id=user_id, limit=limit)['results']
        if not self.multi_anchor:
            return main
        hits = self.anchors.search(query, user_id, limit)
        result = {str(item['id']):item for item in main}
        for hit in hits:
            parent_id = hit['parent_id']
            try:
                point = self._get(parent_id, user_id)
            except ValueError:
                continue
            if point.payload.get('casp_version', 0) != hit['version']:
                continue
            if str(parent_id) not in result:
                result[str(parent_id)] = {'id':parent_id, 'memory':point.payload['data'], 'score':hit['score'], 'user_id':user_id}
        return sorted(result.values(), key=lambda x:x.get('score', 0) or 0, reverse=True)[:limit]

    def add(self, raw_evidence, *, user_id, metadata=None):
        """Mem0 fact/action prompts; execute UPDATE ourselves without swallowed exceptions."""
        if not isinstance(raw_evidence, str) or not raw_evidence.strip() or not user_id:
            raise ValueError('raw_evidence and user_id required')
        from mem0.memory.utils import get_fact_retrieval_messages, parse_messages, remove_code_blocks
        from mem0.configs.prompts import get_update_memory_messages
        m = self.memory
        if getattr(m, 'enable_graph', False):
            raise ValueError('graph mode is not supported by this adapter')
        with self._lock:
            parsed = parse_messages([{'role':'user','content':raw_evidence}])
            custom = m.config.custom_fact_extraction_prompt
            system, prompt = (custom, f'Input:\n{parsed}') if custom else get_fact_retrieval_messages(parsed)
            response = m.llm.generate_response(messages=[{'role':'system','content':system},{'role':'user','content':prompt}], response_format={'type':'json_object'})
            facts = json.loads(remove_code_blocks(response))['facts']
            if not isinstance(facts, list) or any(not isinstance(f,str) or not f.strip() for f in facts):
                raise ValueError('invalid extracted facts')
            if not facts:
                return {'results': []}
            found = {}
            embeddings = {}
            for fact in facts:
                embeddings[fact] = m.embedding_model.embed(fact, 'add')
                for hit in self.search(fact, user_id=user_id, limit=5):
                    found[str(hit['id'])] = hit['memory']
            id_map = {str(i):mid for i,mid in enumerate(found)}
            old = [{'id':i,'text':found[mid]} for i,mid in id_map.items()]
            prompt = get_update_memory_messages(old, facts, m.config.custom_update_memory_prompt)
            response = m.llm.generate_response(messages=[{'role':'user','content':prompt}], response_format={'type':'json_object'})
            actions = json.loads(remove_code_blocks(response))['memory']
            if not isinstance(actions, list):
                raise ValueError('invalid Mem0 actions')
            output = []
            for action in actions:
                try:
                    event, text = action.get('event'), action.get('text')
                    if event == 'ADD':
                        if not isinstance(text,str) or not text.strip(): raise ValueError('empty ADD')
                        payload = {**(metadata or {}), 'user_id':user_id}
                        mid = m._create_memory(text, embeddings, payload)
                        output.append({'id':mid,'event':event,'memory':text})
                    elif event == 'UPDATE':
                        output.append(self.update(id_map[action['id']], raw_evidence=raw_evidence, user_id=user_id))
                    elif event == 'DELETE':
                        output.append(self.delete(id_map[action['id']], user_id=user_id))
                    elif event != 'NONE':
                        raise ValueError(f'unknown Mem0 action {event}')
                except Exception as exc:
                    if output:
                        raise PartialBatchError(output, exc) from exc
                    raise
            return {'results':output}
