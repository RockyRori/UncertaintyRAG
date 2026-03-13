import numpy as np
from utils.text_utils import majority_answer


class DecisionAwareUncertainty:
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def retrieval_uncertainty(self, utilities: list[float]) -> float:
        if not utilities:
            return 1.0
        return float(1.0 - max(utilities))

    def conflict_uncertainty(self, answers: list[str]) -> float:
        cleaned = [a for a in answers if a and a.strip()]
        if not cleaned:
            return 1.0
        _, majority_count = majority_answer(cleaned)
        return float(1.0 - (majority_count / len(cleaned)))

    def stability_uncertainty(self, utilities: list[float], top_m: int = 3) -> float:
        if not utilities:
            return 1.0

        top_utils = sorted(utilities, reverse=True)[:top_m]
        if len(top_utils) == 1:
            return float(1.0 - top_utils[0])

        # 方差越大，说明边界不稳定
        var = float(np.var(top_utils))

        # 简单压到 [0,1] 附近，别让数值乱飞
        return min(var * 4.0, 1.0)

    def total_uncertainty(self, utilities: list[float], answers: list[str]) -> dict:
        ur = self.retrieval_uncertainty(utilities)
        uc = self.conflict_uncertainty(answers)
        us = self.stability_uncertainty(utilities)

        total = self.alpha * ur + self.beta * uc + self.gamma * us
        total = float(min(max(total, 0.0), 1.0))

        return {
            "retrieval_uncertainty": ur,
            "conflict_uncertainty": uc,
            "stability_uncertainty": us,
            "total_uncertainty": total,
        }