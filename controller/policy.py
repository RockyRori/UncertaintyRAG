from controller import state
from decision.actions import RETRIEVE_MORE, RERANK, ANSWER, ABSTAIN, STOP


class RuleBasedPolicy:
    def __init__(
        self,
        tau_answer: float,
        tau_retrieve: float,
        tau_conflict: float,
        tau_stop: float = 0.45,
        tau_delta: float = 0.01,
        tau_gain: float = 0.01,
        answer_min_utility: float = 0.08,
        answer_max_conflict: float = 0.85,
        answer_max_total_uncertainty: float = 0.90,
        tau_stability: float = 0.05,
        retrieve_cost: float = 0.05
    ):
        self.tau_answer = tau_answer
        self.tau_retrieve = tau_retrieve
        self.tau_conflict = tau_conflict
        self.tau_stop = tau_stop
        self.tau_delta = tau_delta
        self.tau_gain = tau_gain

        self.answer_min_utility = answer_min_utility
        self.answer_max_conflict = answer_max_conflict
        self.answer_max_total_uncertainty = answer_max_total_uncertainty
        self.tau_stability = tau_stability
        self.retrieve_cost = retrieve_cost

    def act(self, state) -> str:
        max_u = state.best_utility
        total_u = state.total_uncertainty
        conflict = state.conflict_uncertainty
        stability = state.stability_score
        budget = state.remaining_budget
        delta = state.delta_uncertainty
        gain = state.evidence_gain

        answer_utility = state.answer_utility
        continue_utility = state.continue_utility

        # 1) utility-bounded stopping: 当前回答效用 >= 继续探索效用
        if answer_utility >= continue_utility:
            if (
                max_u >= self.answer_min_utility
                and conflict <= self.answer_max_conflict
                and total_u <= self.answer_max_total_uncertainty
                and stability >= self.tau_stability
            ):
                return ANSWER
            return STOP

        # 2) 预算耗尽，只能停
        if budget <= 0:
            return STOP

        # 3) 如果答案高度冲突，先 rerank/refine
        if conflict > self.tau_conflict and state.last_action != RERANK:
            return RERANK

        # 4) 如果已经检索/重排过，但提升太小，就停止
        if state.last_action in {RETRIEVE_MORE, RERANK}:
            if delta < self.tau_delta and gain < self.tau_gain:
                return STOP

        # 5) retrieval 只有在“高不确定性 + 低utility + 低稳定性”时触发
        if max_u < self.tau_retrieve and total_u > self.tau_stop:
            return RETRIEVE_MORE

        # 6) 否则停止，进入 finalize
        return STOP

    def finalize(self, state) -> str:
        max_u = state.best_utility
        conflict = state.conflict_uncertainty
        total_u = state.total_uncertainty
        stability = state.stability_score

        # 风险受控，允许回答
        if (
            state.best_answer
            and state.answer_utility >= self.answer_min_utility
            and conflict <= self.answer_max_conflict
            and total_u <= self.answer_max_total_uncertainty
        ):
            return ANSWER

        return ABSTAIN