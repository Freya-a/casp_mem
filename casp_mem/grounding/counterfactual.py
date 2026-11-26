def sensitivities(nli, text, evidence, spans, seeds):
    result = {}
    for target in spans:
        if target.id in seeds:
            result[target.id] = 0.0
            continue
        original = ''.join(text[s.start:s.end] for s in spans if s.id != target.id)
        counter = ''.join(evidence if s.id in seeds else text[s.start:s.end] for s in spans if s.id != target.id)
        hypothesis = text[target.start:target.end]
        result[target.id] = max(0.0, nli.score(original, hypothesis)['entailment'] - nli.score(counter, hypothesis)['entailment'])
    return result
