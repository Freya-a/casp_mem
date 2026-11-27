#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import random
import sys
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from casp_mem.config import load_config, project_path


def import_mem0(cfg, internal_dir):
    baseline = project_path(cfg['paths']['mem0'])
    expected = (ROOT / 'mem0_v0.1.118_local').resolve()
    if baseline != expected: raise RuntimeError(f'mem0 path must be copied baseline: {expected}')
    os.environ['MEM0_DIR'] = str(internal_dir)
    os.environ['MEM0_TELEMETRY'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(baseline))
    import mem0
    actual = Path(mem0.__file__).resolve()
    if not actual.is_relative_to(baseline): raise RuntimeError(f'wrong mem0 imported: {actual}')
    return mem0


def build(cfg, data_dir, mem0):
    from casp_mem.clients.llm_client import QwenClient
    from casp_mem.clients.nli_scorer import NLIScorer
    from casp_mem.pipeline import Pipeline
    from casp_mem.integration.mem0_adapter import Mem0Adapter
    models, switches = cfg['models'], cfg['modules']
    for name in ('llm_path','embedding_path','nli_path'):
        if not project_path(models[name]).is_dir(): raise FileNotFoundError(f'{name} missing: {project_path(models[name])}')
    config = {
        'llm': {'provider':'vllm', 'config':{'model':models['llm_served_name'], 'vllm_base_url':models['llm_service_url'], 'api_key':'local-only','temperature':0.0,'max_tokens':1024,'top_p':1.0}},
        'embedder': {'provider':'huggingface', 'config':{'model':str(project_path(models['embedding_path'])), 'embedding_dims':models['embedding_dimensions'], 'model_kwargs':{'device':'cpu','local_files_only':True}}},
        'vector_store': {'provider':'qdrant', 'config':{'collection_name':'memory', 'embedding_model_dims':models['embedding_dimensions'], 'path':str(data_dir / 'memory_qdrant'), 'on_disk':True}},
        'history_db_path':str(data_dir / 'history.sqlite3'),
    }
    memory = mem0.Memory.from_config(config)
    if cfg.get('baseline'): return memory
    llm = QwenClient(models['llm_service_url'], models['llm_served_name'])
    nli = NLIScorer(project_path(models['nli_path']))
    anchors = None
    score_space = None
    if switches.get('multi_anchor', True):
        from qdrant_client import QdrantClient
        from casp_mem.integration.anchor_index import AnchorIndex
        class SharedEmbedder:
            def embed(self, text, query=False):
                return memory.embedding_model.embed(text, 'search' if query else 'add')
        score_space = 'mem0-hf-v1:' + str(project_path(models['embedding_path']))
        anchors = AnchorIndex(QdrantClient(path=str(data_dir / 'anchor_qdrant')), 'anchors', SharedEmbedder(), models['embedding_dimensions'], score_space=score_space)
        from qdrant_client import models as qmodels
        parent_vectors = memory.vector_store.client.get_collection(memory.vector_store.collection_name).config.params.vectors
        anchor_vectors = anchors.client.get_collection(anchors.collection).config.params.vectors
        comparable = (isinstance(parent_vectors, qmodels.VectorParams) and isinstance(anchor_vectors, qmodels.VectorParams) and parent_vectors.distance == anchor_vectors.distance == qmodels.Distance.COSINE)
    else:
        comparable = False
    return Mem0Adapter(memory, Pipeline(llm, nli, switches), anchors, switches.get('multi_anchor', True), comparable_scores=comparable, score_space=score_space)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(ROOT / 'config/casp_mem.yaml'))
    p.add_argument('--check-import', action='store_true')
    p.add_argument('--operation', choices=['add','update','search','delete','recover'], default='add')
    p.add_argument('--user-id')
    p.add_argument('--memory-id')
    p.add_argument('--evidence', help='Original evidence for add/update; query for search')
    args = p.parse_args()
    cfg = load_config(args.config)
    data_dir = project_path(cfg['paths']['results']) / cfg['experiment']
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.check_import:
        mem0 = import_mem0(cfg, data_dir / 'mem0_internal')
        print(Path(mem0.__file__).resolve()); return
    if not args.user_id: p.error('--user-id is required')
    if args.operation in {'update','delete','recover'} and not args.memory_id: p.error('--memory-id is required')
    if args.operation in {'add','update','search'} and not args.evidence: p.error('--evidence is required')
    random.seed(cfg['seed'])
    log = {'started_at':datetime.now(timezone.utc).isoformat(), 'config':cfg, 'seed':cfg['seed'], 'operation':args.operation}
    logfile = data_dir / f'run-{uuid4().hex}.json'
    try:
        mem0 = import_mem0(cfg, data_dir / 'mem0_internal')
        log['mem0_import'] = str(Path(mem0.__file__).resolve())
        obj = build(cfg, data_dir, mem0)
        if cfg.get('baseline'):
            if args.operation == 'add': result = obj.add(args.evidence, user_id=args.user_id)
            elif args.operation == 'search': result = obj.search(args.evidence, user_id=args.user_id)
            else: raise ValueError('baseline CLI supports add/search only')
        elif args.operation == 'add': result = obj.add(args.evidence, user_id=args.user_id)
        elif args.operation == 'update': result = obj.update(args.memory_id, raw_evidence=args.evidence, user_id=args.user_id)
        elif args.operation == 'search': result = obj.search(args.evidence, user_id=args.user_id)
        elif args.operation == 'delete': result = obj.delete(args.memory_id, user_id=args.user_id)
        else: result = obj.recover(args.memory_id, user_id=args.user_id)
        log.update(status='success', result=result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        log.update(status='partial' if exc.__class__.__name__ in {'PartialSyncError','PartialWriteError','PartialBatchError','PartialDeleteError'} else 'failed_or_rejected', error=str(exc), error_type=type(exc).__name__)
        if hasattr(exc, 'completed'):
            log['completed_actions'] = exc.completed
        raise
    finally:
        logfile.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

if __name__ == '__main__': main()
