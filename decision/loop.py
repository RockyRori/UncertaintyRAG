from controller.state import DecisionState
from decision.actions import RETRIEVE_MORE, RERANK, ANSWER, ABSTAIN, STOP
from retrieval.rerank import rerank_by_utility
from utils.text_utils import is_unknown_answer, qa_match


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
        prev_total = state.total_uncertainty
        prev_best = state.best_utility
        prev_best_answer = state.best_answer

        passages = [e["text"] for e in state.evidence]

        if not passages:
            state.utilities = []
            state.candidate_answers = []

            state.generation_entropy = 1.0
            state.utility_uncertainty = 1.0
            state.stability_score = 0.0
            state.total_uncertainty = 1.0

            state.retrieval_uncertainty = 1.0
            state.conflict_uncertainty = 1.0
            state.stability_uncertainty = 1.0

            state.prev_total_uncertainty = prev_total
            state.delta_uncertainty = 0.0

            state.prev_best_utility = prev_best
            state.best_utility = 0.0
            state.evidence_gain = 0.0

            state.prev_best_answer = prev_best_answer
            state.best_answer = ""
            state.best_answer_weight = 0.0
            state.answer_utility = 0.0
            state.continue_utility = 1.0
            state.service_utility = 0.0
            return

        # step 1: per-passage answer
        state.candidate_answers = self.answerer.answer_per_passage(state.question, passages)

        # step 2: utility prediction
        bm25_scores = [float(e.get("score", 0.0)) for e in state.evidence]
        passage_ranks = list(range(1, len(state.evidence) + 1))

        state.utilities = self.utility_predictor.predict_batch(
            question=state.question,
            passages=passages,
            pred_answers=state.candidate_answers,
            bm25_scores=bm25_scores,
            passage_ranks=passage_ranks,
        )

        # step 3: unified uncertainty
        stats = self.uncertainty_scorer.total_uncertainty(
            utilities=state.utilities,
            answers=state.candidate_answers,
            previous_best_answer=prev_best_answer,
        )

        state.generation_entropy = stats["generation_entropy"]
        state.utility_uncertainty = stats["utility_uncertainty"]
        state.stability_score = stats["stability_score"]
        state.total_uncertainty = stats["total_uncertainty"]

        # compatibility fields
        state.retrieval_uncertainty = stats["retrieval_uncertainty"]
        state.conflict_uncertainty = stats["conflict_uncertainty"]
        state.stability_uncertainty = stats["stability_uncertainty"]

        # best answer
        state.prev_best_answer = prev_best_answer
        state.best_answer = stats["best_answer"]
        state.best_answer_weight = stats["best_answer_weight"]

        # scalar utility signals for controller
        state.prev_total_uncertainty = prev_total
        state.delta_uncertainty = max(0.0, prev_total - state.total_uncertainty)

        state.prev_best_utility = prev_best
        state.best_utility = stats.get("best_answer_utility", 0.0)
        state.evidence_gain = max(0.0, state.best_utility - prev_best)

        # utility of answering now
        # 直觉：答案越强、越稳定、越低不确定，utility 越高
        state.answer_utility = (
            0.45 * state.best_utility
            + 0.35 * state.best_answer_weight
            + 0.20 * state.stability_score
        )

        # utility of continuing
        # 直觉：预算越多、当前越不确定、近期提升越明显，越值得继续
        budget_factor = min(1.0, state.remaining_budget / max(1, self.max_budget))
        state.continue_utility = (
            0.35 * state.total_uncertainty
            + 0.25 * budget_factor
            + 0.40 * max(state.delta_uncertainty, state.evidence_gain)
        )

        # service utility (for logging / future extensions)
        state.service_utility = state.answer_utility - 0.1 * (self.max_budget - state.remaining_budget)

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
        if docs:
            state.evidence.extend(docs)

        state.remaining_budget -= 1
        state.last_action = RETRIEVE_MORE

    def _rerank(self, state: DecisionState) -> None:
        if not state.evidence or not state.utilities:
            return

        keep_top_m = min(len(state.evidence), max(2, self.initial_top_k))
        state.evidence, state.utilities = rerank_by_utility(
            state.evidence,
            state.utilities,
            keep_top_m=keep_top_m,
        )

        # rerank 后 candidate_answers 也要同步
        passages = [e["text"] for e in state.evidence]
        state.candidate_answers = self.answerer.answer_per_passage(state.question, passages)

        state.last_action = RERANK

    def _compress_evidence_pool(self, state: DecisionState, keep_top_m: int = 5) -> None:
        if not state.evidence or not state.utilities or not state.candidate_answers:
            return

        ranked = sorted(
            zip(state.evidence, state.utilities, state.candidate_answers),
            key=lambda x: x[1],
            reverse=True,
        )[:keep_top_m]

        state.evidence = [x[0] for x in ranked]
        state.utilities = [x[1] for x in ranked]
        state.candidate_answers = [x[2] for x in ranked]

    def _answer(self, state: DecisionState) -> None:
        passages = [e["text"] for e in state.evidence]
        final_answer = self.answerer.answer(state.question, passages).strip() if passages else ""
        if not final_answer or is_unknown_answer(final_answer):
            final_answer = state.best_answer.strip() if state.best_answer else ""

        if not final_answer or is_unknown_answer(final_answer):
            state.final_answer = "ABSTAIN"
            state.correct = 0
            state.final_action = ABSTAIN
            return

        state.final_answer = final_answer
        state.correct = int(qa_match(final_answer, state.gold_answers))
        state.final_action = ANSWER

    def _abstain(self, state: DecisionState) -> None:
        state.final_answer = "ABSTAIN"
        state.correct = 0
        state.final_action = ABSTAIN

    def _log_step(self, state: DecisionState, action: str) -> None:
        state.history.append(
            {
                "step": state.step,
                "action": action,
                "last_action": state.last_action,
                "remaining_budget": state.remaining_budget,
                "num_evidence": len(state.evidence),

                "best_answer": state.best_answer,
                "best_answer_weight": round(state.best_answer_weight, 4),

                "best_utility": round(state.best_utility, 4),
                "delta_uncertainty": round(state.delta_uncertainty, 4),
                "evidence_gain": round(state.evidence_gain, 4),

                "answer_utility": round(state.answer_utility, 4),
                "continue_utility": round(state.continue_utility, 4),
                "service_utility": round(state.service_utility, 4),

                "utilities": [round(u, 4) for u in state.utilities],
                "candidate_answers": state.candidate_answers,

                "generation_entropy": round(state.generation_entropy, 4),
                "utility_uncertainty": round(state.utility_uncertainty, 4),
                "stability_score": round(state.stability_score, 4),
                "total_uncertainty": round(state.total_uncertainty, 4),

                "retrieval_uncertainty": round(state.retrieval_uncertainty, 4),
                "conflict_uncertainty": round(state.conflict_uncertainty, 4),
                "stability_uncertainty": round(state.stability_uncertainty, 4),
            }
        )

    def run_one(self, question: str, gold_answers: list[str]) -> DecisionState:
        state = DecisionState(
            question=question,
            gold_answers=gold_answers,
            remaining_budget=self.max_budget,
        )

        self._retrieve_initial(state)
        self._update_state_scores(state)

        for step in range(self.max_steps):
            state.step = step + 1

            action = self.policy.act(state)
            self._log_step(state, action)

            if action == RETRIEVE_MORE:
                self._retrieve_more(state)
                self._update_state_scores(state)

                # 控制 evidence pool，防止越检索越嘈杂
                if len(state.evidence) > 5:
                    self._compress_evidence_pool(state, keep_top_m=5)
                    self._update_state_scores(state)
                continue

            if action == RERANK:
                self._rerank(state)
                self._update_state_scores(state)
                continue

            if action == ANSWER:
                self._answer(state)
                return state

            if action == ABSTAIN:
                self._abstain(state)
                return state

            if action == STOP:
                state.stop_reason = "policy_stop"
                final_action = self.policy.finalize(state)
                if final_action == ANSWER:
                    self._answer(state)
                else:
                    self._abstain(state)
                return state

        # max steps reached
        state.stop_reason = "max_steps"
        final_action = self.policy.finalize(state)
        if final_action == ANSWER:
            self._answer(state)
        else:
            self._abstain(state)

        return state
