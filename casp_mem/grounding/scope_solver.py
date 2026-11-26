import itertools
import math

def solve(a, b, edges, seeds, propagation=1.0, max_spans=22):
    """Exact exhaustive binary minimization; reject input too large for bounded work."""
    n = len(a)
    if len(b) != n or n > max_spans or not seeds or not set(seeds) <= set(range(n)):
        raise ValueError('invalid size or no reliable conflict seed')
    if propagation < 0 or any(not math.isfinite(x) or x < 0 for x in [*a, *b, propagation]):
        raise ValueError('invalid costs')
    if any(j not in range(n) or i not in range(n) or not math.isfinite(w) or w < 0 for j, i, w in edges):
        raise ValueError('invalid dependency')
    free = [i for i in range(n) if i not in seeds]
    best, best_set = float('inf'), None
    for bits in itertools.product((0, 1), repeat=len(free)):
        selected = set(seeds) | {i for i, x in zip(free, bits) if x}
        cost = sum(a[i] if i in selected else b[i] for i in range(n))
        cost += propagation * sum(w for j, i, w in edges if j in selected and i not in selected)
        if cost < best:
            best, best_set = cost, selected
    return best_set
