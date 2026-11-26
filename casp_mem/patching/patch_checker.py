def check(nli, evidence, old, patch, threshold):
    # Both directions detect contradictions in classifiers sensitive to ordering.
    return max(nli.score(patch.text, evidence)['contradiction'], nli.score(evidence, patch.text)['contradiction']) < threshold
