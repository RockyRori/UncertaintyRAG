from collections import defaultdict
from typing import List

import numpy as np


class DecisionAwareUncertainty:
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def retrieval_uncertainty(self, utilities: List[float]) -> float:
        if not utilities:
            return 1.0
        max_u = max(float(u) for u in utilities)
        return 1.0 - max_u

    def conflict_uncertainty(self, utilities: List[float], answers: List[str]) -> float:
        if not utilities or not answers or len(utilities) != len(answers):
            return 1.0

        weight_by_answer = defaultdict(float)
        total_weight = 0.0

        for u, ans in zip(utilities, answers):
            ans = str(ans).strip().lower()
            if not ans:
                continue
            w = max(float(u), 0.0)
            weight_by_answer[ans] += w
            total_weight += w

        if total_weight <= 1e-8:
            return 1.0

        majority_weight = max(weight_by_answer.values()) if weight_by_answer else 0.0
        return 1.0 - (majority_weight / total_weight)

    def stability_uncertainty(self, utilities: List[float]) -> float:
        if not utilities:
            return 1.0

        arr = np.array(utilities, dtype=float)

        # 全都一样时方差为 0，表示“排序很稳定”
        var = float(np.var(arr))

        # 做一个轻量裁剪，避免极端值把总分搞飞
        return min(1.0, var)

    def total_uncertainty(self, utilities: List[float], answers: List[str]) -> dict:
        ur = self.retrieval_uncertainty(utilities)
        uc = self.conflict_uncertainty(utilities, answers)
        us = self.stability_uncertainty(utilities)

        total = self.alpha * ur + self.beta * uc + self.gamma * us

        return {
            "retrieval_uncertainty": ur,
            "conflict_uncertainty": uc,
            "stability_uncertainty": us,
            "total_uncertainty": total,
        }