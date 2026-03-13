def rerank_by_utility(evidence: list[dict], utilities: list[float]) -> tuple[list[dict], list[float]]:
    paired = list(zip(evidence, utilities))
    paired.sort(key=lambda x: x[1], reverse=True)

    reranked_evidence = [p[0] for p in paired]
    reranked_utilities = [float(p[1]) for p in paired]
    return reranked_evidence, reranked_utilities