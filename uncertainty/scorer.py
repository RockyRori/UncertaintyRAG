from utils.text_utils import qa_match


class WeakUtilityScorer:
    """
    Phase 2A:
    utility = 1 if single-passage answer is correct else 0
    confidence = max(utilities)
    uncertainty = 1 - confidence
    """

    def score_single_passage_utility(self, pred_answer: str, gold_answers: list[str]) -> int:
        return qa_match(pred_answer, gold_answers)

    def aggregate_confidence(self, utilities: list[int]) -> float:
        if not utilities:
            return 0.0
        return float(max(utilities))

    def aggregate_uncertainty(self, utilities: list[int]) -> float:
        confidence = self.aggregate_confidence(utilities)
        return 1.0 - confidence