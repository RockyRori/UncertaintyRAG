from collections import defaultdict
from typing import List, Dict, Any
import math

from utils.text_utils import is_unknown_answer


class DecisionAwareUncertainty:
    def __init__(self, alpha: float = 0.2, beta: float = 0.5, gamma: float = 0.3):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def _normalize_text(x: str) -> str:
        return str(x).strip().lower()

    def _answer_distribution(self, utilities: List[float], answers: List[str]) -> Dict[str, float]:
        weight_by_answer = defaultdict(float)

        for u, ans in zip(utilities, answers):
            ans_norm = self._normalize_text(ans)
            if not ans_norm or is_unknown_answer(ans_norm):
                continue
            weight_by_answer[ans_norm] += max(float(u), 0.0)

        total = sum(weight_by_answer.values())
        if total <= 1e-12:
            return {}

        return {k: v / total for k, v in weight_by_answer.items()}

    def generation_entropy(self, utilities: List[float], answers: List[str]) -> float:
        """
        工程近似版 H_t:
        用 utility-weighted candidate answer distribution 的熵近似生成熵。
        归一化到 [0,1].
        """
        dist = self._answer_distribution(utilities, answers)
        if not dist:
            return 1.0

        probs = list(dist.values())
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)

        if len(probs) == 1:
            return 0.0

        max_entropy = math.log(len(probs))
        return min(1.0, entropy / (max_entropy + 1e-12))

    def utility_uncertainty(
        self,
        utilities: List[float],
        answers: List[str] | None = None,
    ) -> float:
        """
        1 - u_t, 其中 u_t 近似取当前最优 utility
        """
        if not utilities:
            return 1.0
        if answers is not None:
            utilities = [
                float(u)
                for u, ans in zip(utilities, answers)
                if not is_unknown_answer(ans)
            ]
            if not utilities:
                return 1.0
        u_t = max(float(u) for u in utilities)
        return 1.0 - u_t

    def best_answer_info(self, utilities: List[float], answers: List[str]) -> Dict[str, Any]:
        """
        返回当前 best answer 及其聚合权重
        """
        dist = self._answer_distribution(utilities, answers)
        if not dist:
            return {
                "best_answer": "",
                "best_answer_weight": 0.0,
                "best_answer_utility": 0.0,
            }

        best_answer, best_weight = max(dist.items(), key=lambda x: x[1])
        best_answer_utility = max(
            (
                float(u)
                for u, ans in zip(utilities, answers)
                if self._normalize_text(ans) == best_answer
            ),
            default=0.0,
        )
        return {
            "best_answer": best_answer,
            "best_answer_weight": float(best_weight),
            "best_answer_utility": float(best_answer_utility),
        }

    def stability_score(self, utilities: List[float], answers: List[str], previous_best_answer: str = "") -> float:
        """
        改进版稳定性:
        不再只看“前后答案是否相同”，而是看当前最优答案是否真正占优。
        直觉：
        - top1 和 top2 很接近 -> 不稳定
        - top1 明显领先 -> 稳定
        - 如果前后 best answer 还一致，可以给予少量加分，但不能直接拉满到 1.0
        """
        dist = self._answer_distribution(utilities, answers)
        if not dist:
            return 0.0

        ranked = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        top1_answer, top1_weight = ranked[0]
        top2_weight = ranked[1][1] if len(ranked) > 1 else 0.0

        # 领先优势
        margin = max(0.0, top1_weight - top2_weight)

        # 一致性小加分，但不让它支配一切
        prev = self._normalize_text(previous_best_answer)
        consistency_bonus = 0.15 if prev and top1_answer == prev else 0.0

        score = min(1.0, margin + consistency_bonus)
        return score

    def conflict_uncertainty(self, utilities: List[float], answers: List[str]) -> float:
        """
        兼容旧指标：答案分歧度
        """
        dist = self._answer_distribution(utilities, answers)
        if not dist:
            return 1.0
        majority_weight = max(dist.values())
        return 1.0 - majority_weight

    def total_uncertainty(
        self,
        utilities: List[float],
        answers: List[str],
        previous_best_answer: str = "",
    ) -> Dict[str, Any]:
        h_t = self.generation_entropy(utilities, answers)
        util_u = self.utility_uncertainty(utilities, answers)

        best_info = self.best_answer_info(utilities, answers)
        best_answer = best_info["best_answer"]
        best_answer_weight = best_info["best_answer_weight"]
        best_answer_utility = best_info["best_answer_utility"]

        s_t = self.stability_score(utilities, answers, previous_best_answer)

        total = self.alpha * h_t + self.beta * util_u + self.gamma * (1.0 - s_t)

        # 兼容旧字段
        retrieval_u = util_u
        conflict_u = self.conflict_uncertainty(utilities, answers)
        stability_u = 1.0 - s_t

        return {
            "generation_entropy": h_t,
            "utility_uncertainty": util_u,
            "stability_score": s_t,
            "total_uncertainty": total,
            "best_answer": best_answer,
            "best_answer_weight": best_answer_weight,
            "best_answer_utility": best_answer_utility,
            "retrieval_uncertainty": retrieval_u,
            "conflict_uncertainty": conflict_u,
            "stability_uncertainty": stability_u,
        }
