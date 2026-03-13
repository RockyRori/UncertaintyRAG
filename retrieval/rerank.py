def rerank_by_utility(evidence, utilities, keep_top_m=3, utility_threshold=None):
    paired = list(zip(evidence, utilities))
    paired.sort(key=lambda x: x[1], reverse=True)

    if utility_threshold is not None:
        paired = [p for p in paired if p[1] >= utility_threshold]

    else:
        paired = paired[:keep_top_m]

    new_evidence = [p[0] for p in paired]
    new_utilities = [float(p[1]) for p in paired]

    return new_evidence, new_utilities