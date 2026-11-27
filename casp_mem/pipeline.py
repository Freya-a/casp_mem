from dataclasses import dataclass
from casp_mem.segmentation.segmenter import BOUNDARY, cached_segment, cache_record, segment
from casp_mem.segmentation.preconditions import annotate
from casp_mem.grounding.seed_detector import detect
from casp_mem.grounding.counterfactual import sensitivities
from casp_mem.grounding.scope_solver import solve
from casp_mem.grounding.scope_reviewer import review
from casp_mem.patching.patch_generator import generate
from casp_mem.patching.assembler import assemble
from casp_mem.guard.fidelity_guard import audit

class Rejected(Exception):
    pass

@dataclass
class Result:
    text: str
    patches: list
    anchors: list
    cache: dict
    decisions: list

class Pipeline:
    def __init__(self, llm, nli, options):
        self.llm, self.nli, self.options = llm, nli, options

    def run(self, old, evidence, cache=None, *, forced_seeds=None):
        if not isinstance(old, str) or not old.strip() or not isinstance(evidence, str) or not evidence.strip():
            raise Rejected('old memory and raw evidence are required')
        try:
            enabled = self.options.get('preconditions', True)
            model = getattr(self.llm, 'annotation_cache_key', None)
            annotation_config = {'schema': 1, 'model': model, 'annotator': type(self.llm).__module__ + '.' + type(self.llm).__qualname__, 'preconditions': enabled, 'boundary': BOUNDARY.pattern, 'boundary_flags': BOUNDARY.flags}
            cached = None
            if cache and model and forced_seeds is None:
                try:
                    cached = cached_segment(old, cache, segmenter=lambda _: [], annotation_config=annotation_config)
                except (ValueError, TypeError, AttributeError):
                    pass
            spans = cached if cached is not None else annotate(self.llm, old, segment(old), enabled)
            threshold = self.options.get('contradiction_threshold', .8)
            seeds, scores = detect(self.nli, evidence, old, spans, threshold)
            if forced_seeds is not None:
                forced = set(forced_seeds)
                if not forced or not forced <= {s.id for s in spans}:
                    raise ValueError('invalid forced delete seeds')
                seeds |= forced
            if not seeds:
                raise ValueError('no reliable conflict seed')
            deltas = sensitivities(self.nli, old, evidence, spans, seeds) if self.options.get('counterfactual', True) else {s.id:0.0 for s in spans}
            index = {s.id:i for i,s in enumerate(spans)}
            a = [self.options.get('modification_cost', 0.5)] * len(spans)
            b = [self.options.get('miss_cost', 1.0) * max(scores[s.id], deltas[s.id]) for s in spans]
            weight = self.options.get('dependency_weight', 1.0)
            edges = [(index[s.modifies_id], index[s.id], weight) for s in spans if s.modifies_id is not None]
            selected_idx = solve(a, b, edges, {index[i] for i in seeds}, self.options.get('propagation', 1.0), self.options.get('max_spans', 22))
            selected = {spans[i].id for i in selected_idx}
            decisions = review(self.llm, old, evidence, spans, selected)
            if any(d.label == 'UNCERTAIN' for d in decisions):
                raise ValueError('semantic scope uncertain')
            patches = generate(self.llm, self.nli, old, evidence, spans, decisions, self.options)
            if forced_seeds is not None and any(p.operation != 'PRUNE_OR_ADAPT' or p.op_mode not in {'PRUNE', 'ADAPT'} for p in patches if p.span_id in forced_seeds):
                raise ValueError('partial delete requires a prune or adapt patch')
            locked = {d.span_id for d in decisions if d.label == 'LOCKED_INVARIANT'}
            if forced_seeds is not None and not locked:
                raise ValueError('partial delete has no confirmed retained span')
            candidate, locations = assemble(old, spans, patches, locked)
            # The ablation uses a whitespace-normalized merge, then resegments; it does not guarantee character fidelity.
            # Pruning must never normalize or alter a locked span at a seam.
            hard = self.options.get('hard_lock', True) or any(p.operation == 'PRUNE_OR_ADAPT' and p.op_mode == 'PRUNE' for p in patches)
            if not hard:
                pieces = []
                for s in spans:
                    a0,b0 = locations[s.id]
                    pieces.append(' '.join(candidate[a0:b0].split()))
                candidate = ' '.join(p for p in pieces if p)
                locations = {}
                cursor = 0
                for s, piece in zip(spans, pieces):
                    if piece:
                        start = candidate.find(piece, cursor)
                        locations[s.id] = (start, start + len(piece)); cursor = start + len(piece)
                    else: locations[s.id] = (cursor, cursor)
            audit(self.nli, old, candidate, evidence, spans, locked, locations, threshold, hard_lock=hard, entailment_threshold=self.options.get('entailment_threshold', .5))
            if not candidate.strip():
                raise ValueError('empty candidate requires a confirmed full delete')
            new_cache = None
            if model:
                try:
                    new_cache = cache_record(candidate, annotate(self.llm, candidate, segment(candidate), enabled), annotation_config)
                except Exception:
                    new_cache = None
            return Result(candidate, patches, [a for p in patches for a in p.anchors] if self.options.get('multi_anchor', True) else [], new_cache, decisions)
        except Rejected:
            raise
        except Exception as e:
            raise Rejected(str(e)) from e
