import json
import re
from dataclasses import asdict
from casp_mem.schemas import PatchOp

class QwenClient:
    def __init__(self, base_url, model, api_key='local-only', timeout=120):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model

    def _json(self, instruction, payload, thinking=False):
        response = self.client.chat.completions.create(
            model=self.model, temperature=0, response_format={'type':'json_object'},
            extra_body={'chat_template_kwargs': {'enable_thinking': thinking}},
            messages=[{'role':'system','content':instruction + '\nReturn only one JSON object.'},
                      {'role':'user','content':json.dumps(payload, ensure_ascii=False)}])
        message = response.choices[0].message
        content = message.content or ''
        extra = getattr(message, 'model_extra', None) or {}
        reasoning = extra.get('reasoning_content') or extra.get('reasoning') or getattr(message, 'reasoning_content', None) or ''
        match = re.match(r'^\s*<think>(.*?)</think>\s*', content, flags=re.S)
        if match:
            reasoning = reasoning or match.group(1)
            content = content[match.end():]
        has_reasoning = isinstance(reasoning, str) and bool(reasoning.strip())
        if thinking and not has_reasoning:
            raise ValueError('thinking mode was requested but server returned no observable reasoning; refusing review')
        if not thinking and has_reasoning:
            raise ValueError('non-thinking mode was requested but server returned reasoning')
        return json.loads(content)

    def annotate(self, text, spans, preconditions):
        payload = {'spans':[{'id':s.id, 'text':text[s.start:s.end]} for s in spans], 'preconditions_enabled':preconditions}
        return self._json('Annotate every span with id, kind ASSERTION or CONDITION, scope NONE/LOCAL/FORWARD/GLOBAL, modifies_id (span id or null), and preconditions (1-2 verifiable implicit premises per assertion only if enabled, otherwise []). Never invent a certainty; reject by returning an empty list if ambiguous. Return {"spans": [...]}', payload)['spans']

    def review(self, text, evidence, spans, selected):
        payload = {'evidence':evidence, 'spans':[{'id':s.id, 'text':text[s.start:s.end], 'selected':s.id in selected, 'annotation':asdict(s)} for s in spans]}
        return self._json('Review scope only. For each span output span_id, label and reason. Labels: ATTR_CONFLICT, CONDITION_SHIFT, CASCADED_INVALID, LOCKED_INVARIANT, UNCERTAIN. Selected spans cannot be locked; include any additional dependent spans. Output no rewritten text. Return {"decisions": [...]}', payload, thinking=True)['decisions']

    def patches(self, evidence, context, decisions, split, repair=None):
        payload = {'evidence':evidence,'local_context':context,'decisions':[asdict(d) for d in decisions if any(c['id']==d.span_id for c in context)], 'split_enabled':split, 'repair_instruction':repair}
        rows = self._json('Generate one patch per editable span, no patches for context. Each patch has span_id, operation (REPLACE_ATTR, SPLIT_CONDITION, PRUNE_OR_ADAPT), text, anchors (array), old_branch and new_branch. For SPLIT_CONDITION old_branch must be an exact old text substring, new_branch an exact evidence substring, both at least three characters and included in the text. SPLIT_CONDITION only for explicit source or evidence conditions; each anchor must be an exact source substring and appear in the patch. Preserve punctuation where necessary; if not safe return empty array. Return {"patches": [...]}', payload)['patches']
        return [PatchOp(span_id=p['span_id'], operation=p['operation'], text=p['text'], anchors=tuple(p.get('anchors', [])), old_branch=p.get('old_branch'), new_branch=p.get('new_branch')) for p in rows]
