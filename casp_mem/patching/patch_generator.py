import re
from casp_mem.schemas import PatchOp
from casp_mem.patching.patch_checker import check

OPS = {'REPLACE_ATTR', 'SPLIT_CONDITION', 'PRUNE_OR_ADAPT'}

def generate(llm, nli, text, evidence, spans, decisions, options):
    affected = {d.span_id for d in decisions if d.label != 'LOCKED_INVARIANT'}
    context_ids = set(affected)
    for idx, s in enumerate(spans):
        if s.id in affected:
            context_ids.update(spans[j].id for j in (idx-1, idx+1) if 0 <= j < len(spans))
    context = [{'id':s.id, 'text':text[s.start:s.end], 'editable':s.id in affected} for s in spans if s.id in context_ids]
    def validate(patches):
        if not isinstance(patches, list) or any(not isinstance(p, PatchOp) for p in patches) or {p.span_id for p in patches} != affected or len(patches) != len(affected):
            raise ValueError('patches must cover affected spans exactly once')
        for p in patches:
            decision = next(d for d in decisions if d.span_id == p.span_id)
            if p.operation not in OPS or not isinstance(p.text, str) or (not p.text.strip() and not (p.text == '' and p.operation == 'PRUNE_OR_ADAPT' and p.op_mode == 'PRUNE' and decision.label in {'ATTR_CONFLICT', 'CASCADED_INVALID'} and isinstance(decision.reason, str) and decision.reason.strip())):
                raise ValueError('invalid patch')
            if p.op_mode not in (None, 'PRUNE', 'ADAPT') or (p.op_mode is not None and p.operation != 'PRUNE_OR_ADAPT'):
                raise ValueError('invalid patch mode')
            if decision.label == 'CONDITION_SHIFT' and p.operation != 'SPLIT_CONDITION':
                raise ValueError('condition shift requires a condition split')
            if p.operation == 'SPLIT_CONDITION':
                if not options.get('split_condition', True):
                    raise ValueError('condition splitting disabled')
                if not all(isinstance(x, str) and len(x.strip()) >= 3 for x in (p.old_branch, p.new_branch)) or p.old_branch == p.new_branch or p.old_branch not in text or p.new_branch not in evidence or p.old_branch not in p.text or p.new_branch not in p.text:
                    raise ValueError('condition split requires explicit source-backed old and new branches')
                if decision.label != 'CONDITION_SHIFT':
                    raise ValueError('condition split requires condition-shift review')
                marker = re.compile(r'\b(?:if|when|unless|while|before|after|during|on|at|in|every|since|until|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\d|如果|假如|除非|当|期间|以前|之后|之前|每天|每周|每月|每年|在|时候|上午|下午|晚上|早上|周[一二三四五六日天]|星期', re.I)
                if not p.anchors or any(not isinstance(a, str) or not a.strip() or not marker.search(a) or (a not in text and a not in evidence) or a not in p.text for a in p.anchors):
                    raise ValueError('condition anchors must appear in source and patch')
            elif p.anchors:
                raise ValueError('anchors require a condition split')
    patches = llm.patches(evidence, context, decisions, options.get('split_condition', True))
    validate(patches)
    if options.get('patch_check', True):
        failing = [p for p in patches if p.text and not check(nli, evidence, text, p, options.get('contradiction_threshold', .8))]
        if failing:
            patches = llm.patches(evidence, context, decisions, options.get('split_condition', True), repair='Patch NLI contradiction; repair once')
            validate(patches)
            if any(p.text and not check(nli, evidence, text, p, options.get('contradiction_threshold', .8)) for p in patches):
                raise ValueError('patch NLI check failed after repair')
    return patches
