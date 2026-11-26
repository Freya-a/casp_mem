def detect(nli, evidence, text, spans, threshold):
    scores = {}
    for s in spans:
        targets = [text[s.start:s.end], *s.preconditions]
        comparisons = [nli.score(evidence, target) for target in targets]
        scores[s.id] = max(result['contradiction'] for result in comparisons)
        if s.kind == 'CONDITION':
            scores[s.id] = max(scores[s.id], comparisons[0]['entailment'])
    return {i for i, p in scores.items() if p >= threshold}, scores
