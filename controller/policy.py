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
        answer_min_utility: float = 0.50,
        answer_min_best_utility: float = 0.35,
        answer_max_conflict: float = 0.55,
        answer_max_total_uncertainty: float = 0.70,
        tau_stability: float = 0.12,
        retrieve_cost: float = 0.05,
    ):
        self.tau_answer = tau_answer
        self.tau_retrieve = tau_retrieve
        self.tau_conflict = tau_conflict
        self.tau_stop = tau_stop
        self.tau_delta = tau_delta
        self.tau_gain = tau_gain

        self.answer_min_utility = answer_min_utility
        self.answer_min_best_utility = answer_min_best_utility
        self.answer_max_conflict = answer_max_conflict
        self.answer_max_total_uncertainty = answer_max_total_uncertainty
        self.tau_stability = tau_stability
        self.retrieve_cost = retrieve_cost

    def can_answer(self, state) -> bool:
        return (
            bool(state.best_answer)
            and state.answer_utility >= self.answer_min_utility
            and state.best_utility >= self.answer_min_best_utility
            and state.conflict_uncertainty <= self.answer_max_conflict
            and state.total_uncertainty <= self.answer_max_total_uncertainty
            and state.stability_score >= self.tau_stability
        )

    def act(self, state) -> str:
        max_u = state.best_utility
        total_u = state.total_uncertainty
        conflict = state.conflict_uncertainty
        budget = state.remaining_budget
        delta = state.delta_uncertainty
        gain = state.evidence_gain

        answer_utility = state.answer_utility
        continue_utility = state.continue_utility

        if answer_utility >= continue_utility and self.can_answer(state):
            return ANSWER

        if budget <= 0:
            return STOP

        if conflict > self.tau_conflict and state.last_action != RERANK:
            return RERANK

        if state.last_action in {RETRIEVE_MORE, RERANK}:
            if delta < self.tau_delta and gain < self.tau_gain:
                return STOP

        if max_u < self.tau_retrieve and total_u > self.tau_stop:
            return RETRIEVE_MORE

        return STOP

    def finalize(self, state) -> str:
        if self.can_answer(state):
            return ANSWER

        return ABSTAIN
