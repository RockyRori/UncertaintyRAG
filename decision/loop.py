from controller.state import DecisionState
from decision.actions import RETRIEVE_MORE, RERANK, ANSWER, ABSTAIN, STOP
from retrieval.rerank import rerank_by_utility
from utils.text_utils import qa_match, majority_answer


class DecisionAwareRAG:
    def __init__(
        self,
        retriever,
        utility_predictor,
        answerer,
        uncertainty_scorer,
        policy,
        initial_top_k: int,
        retrieve_more_k: int,
        max_steps: int,
        max_budget: int,
    ):
        self.retriever = retriever
        self.utility_predictor = utility_predictor
        self.answerer = answerer
        self.uncertainty_scorer = uncertainty_scorer
        self.policy = policy
        self.initial_top_k = initial_top_k
        self.retrieve_more_k = retrieve_more_k
        self.max_steps = max_steps
        self.max_budget = max_budget

    def _update_state_scores(self, state: DecisionState) -> None:
        passages = [e["text"] for e in state.evidence]

        if not passages:
            state.utilities = []
            state.candidate_answers = []
            state.retrieval_uncertainty = 1.0
            state.conflict_uncertainty = 1.0
            state.stability_uncertainty = 1.0
            state.total_uncertainty = 1.0
            return

        state.utilities = self.utility_predictor.predict_batch(state.question, passages)
        state.candidate_answers = self.answerer.answer_per_passage(state.question, passages)

        stats = self.uncertainty_scorer.total_uncertainty(
            utilities=state.utilities,
            answers=state.candidate_answers,
        )

        state.retrieval_uncertainty = stats["retrieval_uncertainty"]
        state.conflict_uncertainty = stats["conflict_uncertainty"]
        state.stability_uncertainty = stats["stability_uncertainty"]
        state.total_uncertainty = stats["total_uncertainty"]

    def _retrieve_initial(self, state: DecisionState) -> None:
        docs = self.retriever.retrieve(
            question=state.question,
            top_k=self.initial_top_k,
            offset=0,
            exclude_ids=set(),
        )
        state.evidence.extend(docs)

    def _retrieve_more(self, state: DecisionState) -> None:
        existing_ids = {e["id"] for e in state.evidence}
        docs = self.retriever.retrieve(
            question=state.question,
            top_k=self.retrieve_more_k,
            offset=0,
            exclude_ids=existing_ids,
        )
        state.evidence.extend(docs)
        state.remaining_budget -= 1

    def _rerank(self, state: DecisionState) -> None:
        if not state.evidence or not state.utilities:
            return
        state.evidence, state.utilities = rerank_by_utility(state.evidence, state.utilities)

    def _answer(self, state: DecisionState) -> None:
        top_passages = [e["text"] for e in state.evidence[:3]]
        pred = self.answerer.answer(state.question, top_passages)

        # 如果多 passage 生成很飘，也可以 fallback 到多数投票的单 passage answer
        maj_ans, maj_count = majority_answer(state.candidate_answers[:3])
        if not pred or not pred.strip():
            pred = maj_ans

        state.final_answer = pred
        state.correct = qa_match(pred, state.gold_answers)
        state.final_action = ANSWER

    def _abstain(self, state: DecisionState) -> None:
        state.final_answer = "ABSTAIN"
        state.correct = 0
        state.final_action = ABSTAIN

    def run_one(self, question: str, gold_answers: list[str]) -> DecisionState:
        state = DecisionState(
            question=question,
            gold_answers=gold_answers,
            remaining_budget=self.max_budget,
        )

        self._retrieve_initial(state)

        for step in range(self.max_steps):
            state.step = step + 1
            self._update_state_scores(state)

            action = self.policy.act(state)

            state.history.append({
                "step": state.step,
                "action": action,
                "remaining_budget": state.remaining_budget,
                "num_evidence": len(state.evidence),
                "utilities": [round(u, 4) for u in state.utilities],
                "candidate_answers": state.candidate_answers,
                "retrieval_uncertainty": round(state.retrieval_uncertainty, 4),
                "conflict_uncertainty": round(state.conflict_uncertainty, 4),
                "stability_uncertainty": round(state.stability_uncertainty, 4),
                "total_uncertainty": round(state.total_uncertainty, 4),
            })

            if action == RETRIEVE_MORE:
                self._retrieve_more(state)
                continue

            if action == RERANK:
                self._rerank(state)
                # rerank 后再走一轮
                continue

            if action == ANSWER:
                self._answer(state)
                return state

            if action == ABSTAIN:
                self._abstain(state)
                return state

        # max_steps 到了还没出结论
        if state.utilities and max(state.utilities) >= self.policy.tau_answer:
            self._answer(state)
        else:
            self._abstain(state)

        return state