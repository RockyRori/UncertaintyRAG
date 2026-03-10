class UncertaintyScorer:
    def retrieval_uncertainty(self, retrieved_docs: list[dict]) -> float:
        if not retrieved_docs:
            return 1.0

        scores = [doc["score"] for doc in retrieved_docs]

        if len(scores) == 1:
            return 0.5

        top1 = scores[0]
        top2 = scores[1]

        margin = top1 - top2

        # margin 越小，不确定性越高
        uncertainty = 1.0 / (1.0 + max(margin, 1e-6))
        return float(uncertainty)