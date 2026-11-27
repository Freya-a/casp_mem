"""Resolve inferred deletes without changing the native Mem0 implementation."""
from casp_mem.pipeline import Rejected
from casp_mem.segmentation.segmenter import segment


class DeleteGuard:
    def __init__(self, adapter):
        self.adapter = adapter

    def handle(self, memory_id, raw_evidence, user_id):
        old = self.adapter._get(memory_id, user_id).payload['data']
        try:
            verdict = self.adapter.pipeline.llm.classify_delete(old, raw_evidence)
            if not isinstance(verdict, dict) or not isinstance(verdict.get('reason'), str) or not verdict['reason'].strip():
                raise ValueError('invalid delete assessment')
            decision = verdict.get('decision')
            if decision not in {'FULL', 'PRUNE', 'UNCERTAIN'}:
                raise ValueError('invalid delete assessment decision')
            ids = verdict.get('invalid_span_ids')
            if decision == 'PRUNE':
                valid_ids = {s.id for s in segment(old)}
                if not isinstance(ids, list) or not ids or any(type(i) is not int or i not in valid_ids for i in ids) or len(set(ids)) != len(ids) or set(ids) == valid_ids:
                    raise ValueError('partial delete scope is ambiguous')
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            return {'id': memory_id, 'event': 'RETAIN', 'memory': old, 'reason': str(exc)}
        if decision == 'FULL' and verdict.get('whole_confirmed') is True:
            return self.adapter.delete(memory_id, user_id=user_id)
        if decision == 'PRUNE':
            try:
                result = self.adapter.update(memory_id, raw_evidence=raw_evidence, user_id=user_id, forced_seeds=ids)
            except Rejected as exc:
                return {'id': memory_id, 'event': 'RETAIN', 'memory': old, 'reason': str(exc)}
            result['delete_resolution'] = 'PRUNED_OR_ADAPTED'
            return result
        return {'id': memory_id, 'event': 'RETAIN', 'memory': old, 'reason': verdict['reason']}
