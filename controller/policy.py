from decision.actions import RETRIEVE_MORE, RERANK, ANSWER, ABSTAIN


class RuleBasedPolicy:
    def __init__(self, tau_answer: float, tau_retrieve: float, tau_conflict: float):
        self.tau_answer = tau_answer
        self.tau_retrieve = tau_retrieve
        self.tau_conflict = tau_conflict

    def act(self, state) -> str:
        max_u = max(state.utilities) if state.utilities else 0.0
        conflict = state.conflict_uncertainty
        budget = state.remaining_budget

        if max_u >= self.tau_answer and conflict <= self.tau_conflict:
            return ANSWER

        if max_u < self.tau_retrieve and budget > 0:
            return RETRIEVE_MORE

        if budget == 0 and max_u < self.tau_answer:
            return ABSTAIN

        return RERANK