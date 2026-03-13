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
        answer_min_utility: float = 0.14,
        answer_max_conflict: float = 0.70,
        answer_max_total_uncertainty: float = 0.65,
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

    def act(self, state) -> str:
        max_u = max(state.utilities) if state.utilities else 0.0
        total_u = state.total_uncertainty
        conflict = state.conflict_uncertainty
        budget = state.remaining_budget
        delta = state.delta_uncertainty
        gain = state.evidence_gain

        # 1) 已经足够好，直接答
        if max_u >= self.tau_answer and conflict <= self.tau_conflict:
            return ANSWER

        # 2) 已经不算特别不确定，直接停，交给 finalize 判断
        if total_u <= self.tau_stop:
            return STOP

        # 3) 如果有明显冲突，优先 rerank，而不是盲目继续找
        if conflict > self.tau_conflict and state.last_action != RERANK:
            return RERANK

        # 4) 如果上一步已经检索/重排过，但收益很小，就停
        if state.last_action in {RETRIEVE_MORE, RERANK}:
            if delta < self.tau_delta and gain < self.tau_gain:
                return STOP

        # 5) 预算没了，只能停
        if budget <= 0:
            return STOP

        # 6) 只有 utility 真的偏低时，才继续检索
        if max_u < self.tau_retrieve:
            return RETRIEVE_MORE

        # 7) 默认停，不要瞎找
        return STOP

    def finalize(self, state) -> str:
        max_u = max(state.utilities) if state.utilities else 0.0
        conflict = state.conflict_uncertainty
        total_u = state.total_uncertainty

        # 高置信直接答
        if max_u >= self.tau_answer and conflict <= self.tau_conflict:
            return ANSWER

        # 总体已经足够稳，也答
        if total_u <= self.tau_stop:
            return ANSWER

        # 参数化的 decision boundary
        if (
            state.candidate_answers
            and max_u >= self.answer_min_utility
            and conflict <= self.answer_max_conflict
            and total_u <= self.answer_max_total_uncertainty
        ):
            return ANSWER

        return ABSTAIN
