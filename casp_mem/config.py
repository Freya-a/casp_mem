from copy import deepcopy
from pathlib import Path
import re
import yaml

ROOT = Path(__file__).resolve().parent.parent

def load_config(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen: raise ValueError('cyclic config inheritance')
    seen.add(path)
    with path.open(encoding='utf-8') as f: data = yaml.safe_load(f)
    if not isinstance(data, dict): raise ValueError('config must be a mapping')
    base = load_config(path.parent / data.pop('extends'), seen) if 'extends' in data else {}
    def merge(a, b):
        a = deepcopy(a)
        for key, val in b.items():
            a[key] = merge(a.get(key, {}), val) if isinstance(val, dict) and isinstance(a.get(key, {}), dict) else val
        return a
    cfg = merge(base, data)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', cfg['experiment']): raise ValueError('unsafe experiment name')
    return cfg

def project_path(value):
    p = Path(value)
    if p.is_absolute(): raise ValueError('paths must be relative to project root')
    resolved = (ROOT / p).resolve()
    if not resolved.is_relative_to(ROOT): raise ValueError('path escapes project root')
    return resolved
