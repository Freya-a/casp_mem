def audit(nli, old, candidate, evidence, spans, locked, locations, threshold=.8, window=120, hard_lock=True, entailment_threshold=.5):
    for s in spans:
        if s.id not in locked:
            continue
        start, end = locations[s.id]
        original = old[s.start:s.end]
        if hard_lock and candidate[start:end] != original:
            raise ValueError('locked characters changed')
        local = candidate[max(0,start-window):min(len(candidate),end+window)]
        local_score = nli.score(local, original)
        if local_score['contradiction'] >= threshold or local_score['entailment'] < entailment_threshold:
            raise ValueError('locked span fails local window audit')
    if max(nli.score(candidate, evidence)['contradiction'], nli.score(evidence, candidate)['contradiction']) >= threshold:
        raise ValueError('candidate conflicts with evidence')
