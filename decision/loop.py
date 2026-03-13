from controller.state import DecisionState
from decision.actions import RETRIEVE_MORE, RERANK, ANSWER, ABSTAIN, STOP
from retrieval.rerank import rerank_by_utility
from utils.text_utils import qa_match, majority_answer
from collections import defaultdict


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
        prev_best = max(state.utilities) if state.utilities else 0.0

        passages = [e["text"] for e in state.evidence]

        if not passages:
            state.utilities = []
            state.candidate_answers = []
            state.retrieval_uncertainty = 1.0
            state.conflict_uncertainty = 1.0
            state.stability_uncertainty = 1.0
            state.total_uncertainty = 1.0
            state.prev_total_uncertainty = prev_total
            state.delta_uncertainty = 0.0
            state.prev_best_utility = prev_best
            state.best_utility = 0.0
            state.evidence_gain = 0.0
            return

        # 先逐 passage 生成答案
        state.candidate_answers = self.answerer.answer_per_passage(state.question, passages)

        bm25_scores = [float(e.get("score", 0.0)) for e in state.evidence]
        passage_ranks = list(range(1, len(state.evidence) + 1))

        # 再结合 question + passage + pred_answer + structured features 预测 utility
        state.utilities = self.utility_predictor.predict_batch(
            question=state.question,
            passages=passages,
            pred_answers=state.candidate_answers,
            bm25_scores=bm25_scores,
            passage_ranks=passage_ranks,
        )

        stats = self.uncertainty_scorer.total_uncertainty(
            utilities=state.utilities,
            answers=state.candidate_answers,
        )

        state.retrieval_uncertainty = stats["retrieval_uncertainty"]
        state.conflict_uncertainty = stats["conflict_uncertainty"]
        state.stability_uncertainty = stats["stability_uncertainty"]
        state.total_uncertainty = stats["total_uncertainty"]

        state.prev_total_uncertainty = prev_total
        state.delta_uncertainty = max(0.0, prev_total - state.total_uncertainty)

        state.prev_best_utility = prev_best
        state.best_utility = max(state.utilities) if state.utilities else 0.0
        state.evidence_gain = max(0.0, state.best_utility - prev_best)

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
        state.last_action = RERANK

    def _answer(self, state: DecisionState) -> None:
        pred = ""

        if state.utilities and state.candidate_answers:
            answer_scores = defaultdict(float)

            for u, ans in zip(state.utilities, state.candidate_answers):
                ans = str(ans).strip()
                if ans:
                    answer_scores[ans] += float(u)

            if answer_scores:
                pred = max(answer_scores.items(), key=lambda x: x[1])[0]

        if not pred and state.candidate_answers:
            maj_ans, _ = majority_answer(state.candidate_answers[:3])
            pred = maj_ans

        if not pred or not str(pred).strip():
            top_passages = [e["text"] for e in state.evidence[:3]]
            pred = self.answerer.answer(state.question, top_passages)

        state.final_answer = pred if pred else "ABSTAIN"
        state.correct = qa_match(state.final_answer, state.gold_answers)
        state.final_action = ANSWER

    def _abstain(self, state: DecisionState) -> None:
        state.final_answer = "ABSTAIN"
        state.correct = 0
        state.final_action = ABSTAIN

    def _log_step(self, state: DecisionState, action: str) -> None:
        state.history.append({
            "step": state.step,
            "action": action,
            "last_action": state.last_action,
            "remaining_budget": state.remaining_budget,
            "num_evidence": len(state.evidence),
            "best_utility": round(state.best_utility, 4),
            "delta_uncertainty": round(state.delta_uncertainty, 4),
            "evidence_gain": round(state.evidence_gain, 4),
            "utilities": [round(u, 4) for u in state.utilities],
            "candidate_answers": state.candidate_answers,
            "retrieval_uncertainty": round(state.retrieval_uncertainty, 4),
            "conflict_uncertainty": round(state.conflict_uncertainty, 4),
            "stability_uncertainty": round(state.stability_uncertainty, 4),
            "total_uncertainty": round(state.total_uncertainty, 4),
        })

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
                        # 控制证据池规模，避免越检索越嘈杂
                if len(state.evidence) > 5 and state.utilities:
                    ranked = sorted(
                        zip(state.evidence, state.utilities, state.candidate_answers),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]

                    state.evidence = [x[0] for x in ranked]
                    state.utilities = [x[1] for x in ranked]
                    state.candidate_answers = [x[2] for x in ranked]

                    # 截断后重新计算 uncertainty
                    stats = self.uncertainty_scorer.total_uncertainty(
                        utilities=state.utilities,
                        answers=state.candidate_answers,
                    )
                    state.retrieval_uncertainty = stats["retrieval_uncertainty"]
                    state.conflict_uncertainty = stats["conflict_uncertainty"]
                    state.stability_uncertainty = stats["stability_uncertainty"]
                    state.total_uncertainty = stats["total_uncertainty"]
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