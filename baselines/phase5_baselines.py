from typing import Any, Dict, List

from retrieval.rerank import rerank_by_utility
from utils.text_utils import qa_match, majority_answer


def _build_record(
    question: str,
    gold_answers: List[str],
    final_answer: str,
    final_action: str,
    correct: int,
    uncertainty: float,
    retrieval_uncertainty: float,
    conflict_uncertainty: float,
    stability_uncertainty: float,
    steps: int,
    num_evidence: int,
    budget_used: int,
    history: List[Dict[str, Any]] | None = None,
    stop_reason: str = "",
) -> Dict[str, Any]:
    return {
        "question": question,
        "gold_answers": gold_answers,
        "final_answer": final_answer,
        "final_action": final_action,
        "correct": correct,
        "uncertainty": uncertainty,
        "retrieval_uncertainty": retrieval_uncertainty,
        "conflict_uncertainty": conflict_uncertainty,
        "stability_uncertainty": stability_uncertainty,
        "steps": steps,
        "num_evidence": num_evidence,
        "budget_used": budget_used,
        "stop_reason": stop_reason,
        "history": history or [],
    }


def _compute_signals(question, evidence, utility_predictor, answerer, uncertainty_scorer):
    passages = [e["text"] for e in evidence]

    if not passages:
        return {
            "utilities": [],
            "candidate_answers": [],
            "retrieval_uncertainty": 1.0,
            "conflict_uncertainty": 1.0,
            "stability_uncertainty": 1.0,
            "total_uncertainty": 1.0,
        }

    candidate_answers = answerer.answer_per_passage(question, passages)
    bm25_scores = [float(e.get("score", 0.0)) for e in evidence]
    passage_ranks = list(range(1, len(evidence) + 1))

    utilities = utility_predictor.predict_batch(
        question=question,
        passages=passages,
        pred_answers=candidate_answers,
        bm25_scores=bm25_scores,
        passage_ranks=passage_ranks,
    )

    stats = uncertainty_scorer.total_uncertainty(
        utilities=utilities,
        answers=candidate_answers,
    )

    return {
        "utilities": utilities,
        "candidate_answers": candidate_answers,
        "retrieval_uncertainty": stats["retrieval_uncertainty"],
        "conflict_uncertainty": stats["conflict_uncertainty"],
        "stability_uncertainty": stats["stability_uncertainty"],
        "total_uncertainty": stats["total_uncertainty"],
    }


def _final_answer_from_evidence(question, evidence, candidate_answers, answerer, top_n: int = 3) -> str:
    top_passages = [e["text"] for e in evidence[:top_n]]
    pred = answerer.answer(question, top_passages)

    maj_ans, _ = majority_answer(candidate_answers[:top_n])
    if not pred or not pred.strip():
        pred = maj_ans

    return pred if pred else "ABSTAIN"


def run_single_shot(
    question: str,
    gold_answers: List[str],
    retriever,
    utility_predictor,
    answerer,
    uncertainty_scorer,
    top_k: int = 3,
) -> Dict[str, Any]:
    evidence = retriever.retrieve(
        question=question,
        top_k=top_k,
        offset=0,
        exclude_ids=set(),
    )

    signals = _compute_signals(
        question=question,
        evidence=evidence,
        utility_predictor=utility_predictor,
        answerer=answerer,
        uncertainty_scorer=uncertainty_scorer,
    )

    pred = _final_answer_from_evidence(
        question=question,
        evidence=evidence,
        candidate_answers=signals["candidate_answers"],
        answerer=answerer,
    )
    correct = qa_match(pred, gold_answers)

    history = [{
        "step": 1,
        "action": "SINGLE_SHOT_ANSWER",
        "remaining_budget": 0,
        "num_evidence": len(evidence),
        "utilities": [round(u, 4) for u in signals["utilities"]],
        "candidate_answers": signals["candidate_answers"],
        "retrieval_uncertainty": round(signals["retrieval_uncertainty"], 4),
        "conflict_uncertainty": round(signals["conflict_uncertainty"], 4),
        "stability_uncertainty": round(signals["stability_uncertainty"], 4),
        "total_uncertainty": round(signals["total_uncertainty"], 4),
    }]

    return _build_record(
        question=question,
        gold_answers=gold_answers,
        final_answer=pred,
        final_action="ANSWER",
        correct=correct,
        uncertainty=signals["total_uncertainty"],
        retrieval_uncertainty=signals["retrieval_uncertainty"],
        conflict_uncertainty=signals["conflict_uncertainty"],
        stability_uncertainty=signals["stability_uncertainty"],
        steps=1,
        num_evidence=len(evidence),
        budget_used=0,
        history=history,
        stop_reason="single_shot",
    )


def run_single_shot_rerank(
    question: str,
    gold_answers: List[str],
    retriever,
    utility_predictor,
    answerer,
    uncertainty_scorer,
    top_k: int = 5,
    keep_top_m: int = 3,
) -> Dict[str, Any]:
    evidence = retriever.retrieve(
        question=question,
        top_k=top_k,
        offset=0,
        exclude_ids=set(),
    )

    initial_signals = _compute_signals(
        question=question,
        evidence=evidence,
        utility_predictor=utility_predictor,
        answerer=answerer,
        uncertainty_scorer=uncertainty_scorer,
    )

    reranked_evidence, reranked_utilities = rerank_by_utility(
        evidence,
        initial_signals["utilities"],
        keep_top_m=min(keep_top_m, len(evidence)),
    )

    # rerank 后重新算 candidate answers 和 uncertainty
    signals = _compute_signals(
        question=question,
        evidence=reranked_evidence,
        utility_predictor=utility_predictor,
        answerer=answerer,
        uncertainty_scorer=uncertainty_scorer,
    )

    pred = _final_answer_from_evidence(
        question=question,
        evidence=reranked_evidence,
        candidate_answers=signals["candidate_answers"],
        answerer=answerer,
    )
    correct = qa_match(pred, gold_answers)

    history = [
        {
            "step": 1,
            "action": "RETRIEVE",
            "remaining_budget": 0,
            "num_evidence": len(evidence),
            "utilities": [round(u, 4) for u in initial_signals["utilities"]],
            "candidate_answers": initial_signals["candidate_answers"],
            "retrieval_uncertainty": round(initial_signals["retrieval_uncertainty"], 4),
            "conflict_uncertainty": round(initial_signals["conflict_uncertainty"], 4),
            "stability_uncertainty": round(initial_signals["stability_uncertainty"], 4),
            "total_uncertainty": round(initial_signals["total_uncertainty"], 4),
        },
        {
            "step": 2,
            "action": "RERANK_ANSWER",
            "remaining_budget": 0,
            "num_evidence": len(reranked_evidence),
            "utilities": [round(u, 4) for u in signals["utilities"]],
            "candidate_answers": signals["candidate_answers"],
            "retrieval_uncertainty": round(signals["retrieval_uncertainty"], 4),
            "conflict_uncertainty": round(signals["conflict_uncertainty"], 4),
            "stability_uncertainty": round(signals["stability_uncertainty"], 4),
            "total_uncertainty": round(signals["total_uncertainty"], 4),
        },
    ]

    return _build_record(
        question=question,
        gold_answers=gold_answers,
        final_answer=pred,
        final_action="ANSWER",
        correct=correct,
        uncertainty=signals["total_uncertainty"],
        retrieval_uncertainty=signals["retrieval_uncertainty"],
        conflict_uncertainty=signals["conflict_uncertainty"],
        stability_uncertainty=signals["stability_uncertainty"],
        steps=2,
        num_evidence=len(reranked_evidence),
        budget_used=0,
        history=history,
        stop_reason="single_shot_rerank",
    )


def run_single_shot_abstain(
    question: str,
    gold_answers: List[str],
    retriever,
    utility_predictor,
    answerer,
    uncertainty_scorer,
    top_k: int = 3,
    tau_answer: float = 0.75,
    tau_conflict: float = 0.35,
    tau_stop: float = 0.30,
) -> Dict[str, Any]:
    evidence = retriever.retrieve(
        question=question,
        top_k=top_k,
        offset=0,
        exclude_ids=set(),
    )

    signals = _compute_signals(
        question=question,
        evidence=evidence,
        utility_predictor=utility_predictor,
        answerer=answerer,
        uncertainty_scorer=uncertainty_scorer,
    )

    max_u = max(signals["utilities"]) if signals["utilities"] else 0.0
    conflict = signals["conflict_uncertainty"]
    total_u = signals["total_uncertainty"]

    should_answer = (max_u >= tau_answer and conflict <= tau_conflict) or (total_u <= tau_stop)

    if should_answer:
        pred = _final_answer_from_evidence(
            question=question,
            evidence=evidence,
            candidate_answers=signals["candidate_answers"],
            answerer=answerer,
        )
        final_action = "ANSWER"
        correct = qa_match(pred, gold_answers)
        stop_reason = "single_shot_answer"
    else:
        pred = "ABSTAIN"
        final_action = "ABSTAIN"
        correct = 0
        stop_reason = "single_shot_abstain"

    history = [{
        "step": 1,
        "action": final_action,
        "remaining_budget": 0,
        "num_evidence": len(evidence),
        "utilities": [round(u, 4) for u in signals["utilities"]],
        "candidate_answers": signals["candidate_answers"],
        "retrieval_uncertainty": round(signals["retrieval_uncertainty"], 4),
        "conflict_uncertainty": round(signals["conflict_uncertainty"], 4),
        "stability_uncertainty": round(signals["stability_uncertainty"], 4),
        "total_uncertainty": round(signals["total_uncertainty"], 4),
    }]

    return _build_record(
        question=question,
        gold_answers=gold_answers,
        final_answer=pred,
        final_action=final_action,
        correct=correct,
        uncertainty=signals["total_uncertainty"],
        retrieval_uncertainty=signals["retrieval_uncertainty"],
        conflict_uncertainty=signals["conflict_uncertainty"],
        stability_uncertainty=signals["stability_uncertainty"],
        steps=1,
        num_evidence=len(evidence),
        budget_used=0,
        history=history,
        stop_reason=stop_reason,
    )


def run_decision_loop(
    question: str,
    gold_answers: List[str],
    runner,
) -> Dict[str, Any]:
    state = runner.run_one(question=question, gold_answers=gold_answers)

    return _build_record(
        question=question,
        gold_answers=gold_answers,
        final_answer=state.final_answer,
        final_action=state.final_action,
        correct=state.correct,
        uncertainty=state.total_uncertainty,
        retrieval_uncertainty=state.retrieval_uncertainty,
        conflict_uncertainty=state.conflict_uncertainty,
        stability_uncertainty=state.stability_uncertainty,
        steps=state.step,
        num_evidence=len(state.evidence),
        budget_used=runner.max_budget - state.remaining_budget,
        history=state.history,
        stop_reason=state.stop_reason,
    )