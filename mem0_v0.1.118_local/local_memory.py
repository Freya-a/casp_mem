"""Local-only configuration for the unchanged Mem0 memory implementation."""
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT.parent / 'model'


def build_config(data_dir, base_url='http://127.0.0.1:18888/v1'):
    url = urlparse(base_url)
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1') or url.username or url.password or url.path.rstrip('/') != '/v1':
        raise ValueError('The language model must use a loopback HTTP /v1 endpoint')
    data = Path(data_dir).expanduser().resolve()
    return {
        'llm': {'provider': 'vllm', 'config': {
            'model': 'Qwen3-8B-local', 'vllm_base_url': base_url,
            'api_key': 'local-only', 'temperature': 0.0, 'max_tokens': 1024,
            'top_p': 1.0,
        }},
        'embedder': {'provider': 'huggingface', 'config': {
            'model': str(MODEL_ROOT / 'Qwen3-Embedding-0.6B'),
            'embedding_dims': 1024,
            'model_kwargs': {'device': 'cpu', 'local_files_only': True},
        }},
        'vector_store': {'provider': 'qdrant', 'config': {
            'collection_name': 'mem0_local', 'embedding_model_dims': 1024,
            'path': str(data / 'qdrant'), 'on_disk': True,
        }},
        'history_db_path': str(data / 'history.sqlite3'),
    }


def create_memory(data_dir=None):
    data = Path(data_dir or ROOT / 'local_data').expanduser().resolve()
    data.mkdir(parents=True, exist_ok=True)
    local_mem0_dir = str(data / 'mem0_internal')
    if 'mem0.memory.setup' in sys.modules and sys.modules['mem0.memory.setup'].mem0_dir != local_mem0_dir:
        raise RuntimeError('Mem0 was already imported with a different MEM0_DIR; start a fresh process')
    os.environ['MEM0_DIR'] = local_mem0_dir
    os.environ['MEM0_TELEMETRY'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['NO_PROXY'] = '127.0.0.1,localhost,::1'
    os.environ['no_proxy'] = os.environ['NO_PROXY']
    from mem0 import Memory
    if not Path(sys.modules['mem0'].__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError('Wrong mem0 package imported; run with this directory first on PYTHONPATH')
    return Memory.from_config(build_config(data))
