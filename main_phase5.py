from config import (
    CORPUS_PATH,
    MINI_DATASET_PATH,
    OUTPUTS_DIR,
    INITIAL_TOP_K,
    RETRIEVE_MORE_K,
    MAX_DECISION_STEPS,
    MAX_RETRIEVAL_BUDGET,
    TAU_ANSWER,
    TAU_RETRIEVE,
    TAU_CONFLICT,
    TAU_STOP,
    TAU_DELTA,
    TAU_GAIN,
    UNCERTAINTY_ALPHA,
    UNCERTAINTY_BETA,
    UNCERTAINTY_GAMMA,
    GENERATOR_MODEL_NAME,
    MAX_INPUT_LENGTH,
    MAX_NEW_TOKENS,
)
from utils.io_utils import load_json, save_json
from retriever.bm25_retriever import BM25Retriever
from inference.predict_utility import UtilityPredictor
from generator.simple_answerer import SimpleAnswerer
from uncertainty.signals import DecisionAwareUncertainty
from controller.policy import RuleBasedPolicy
from decision.loop import DecisionAwareRAG
from evaluation.metrics import (
    compute_accuracy,
    compute_auroc,
    compute_avg_uncertainty,
    selective_accuracy,
)
from evaluation.decision_metrics import summarize_decision_records


def main():
    dataset = load_json(MINI_DATASET_PATH)

    retriever = BM25Retriever(CORPUS_PATH)
    utility_predictor = UtilityPredictor()
    answerer = SimpleAnswerer(
        model_name=GENERATOR_MODEL_NAME,
        max_input_length=MAX_INPUT_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    uncertainty_scorer = DecisionAwareUncertainty(
        alpha=UNCERTAINTY_ALPHA,
        beta=UNCERTAINTY_BETA,
        gamma=UNCERTAINTY_GAMMA,
    )
    policy = RuleBasedPolicy(
        tau_answer=TAU_ANSWER,
        tau_retrieve=TAU_RETRIEVE,
        tau_conflict=TAU_CONFLICT,
        tau_stop=TAU_STOP,
        tau_delta=TAU_DELTA,
        tau_gain=TAU_GAIN,
    )

    runner = DecisionAwareRAG(
        retriever=retriever,
        utility_predictor=utility_predictor,
        answerer=answerer,
        uncertainty_scorer=uncertainty_scorer,
        policy=policy,
        initial_top_k=INITIAL_TOP_K,
        retrieve_more_k=RETRIEVE_MORE_K,
        max_steps=MAX_DECISION_STEPS,
        max_budget=MAX_RETRIEVAL_BUDGET,
    )

    records = []

    for sample in dataset[:10]:
        question = sample["question"]
        gold_answers = sample["gold_answers"]

        state = runner.run_one(question=question, gold_answers=gold_answers)

        records.append(
            {
                "question": question,
                "gold_answers": gold_answers,
                "final_answer": state.final_answer,
                "final_action": state.final_action,
                "correct": state.correct,
                "uncertainty": state.total_uncertainty,
                "retrieval_uncertainty": state.retrieval_uncertainty,
                "conflict_uncertainty": state.conflict_uncertainty,
                "stability_uncertainty": state.stability_uncertainty,
                "steps": state.step,
                "num_evidence": len(state.evidence),
                "budget_used": MAX_RETRIEVAL_BUDGET - state.remaining_budget,
                "stop_reason": state.stop_reason,
                "history": state.history,
            }
        )

        print("=" * 80)
        print(f"Question: {question}")
        print(f"Gold: {gold_answers}")
        print(f"Action: {state.final_action}")
        print(f"Pred: {state.final_answer}")
        print(f"Correct: {state.correct}")
        print(f"Retrieval U: {state.retrieval_uncertainty:.4f}")
        print(f"Conflict U : {state.conflict_uncertainty:.4f}")
        print(f"Stability U: {state.stability_uncertainty:.4f}")
        print(f"Total U    : {state.total_uncertainty:.4f}")
        print(f"Steps: {state.step}")
        print(f"Evidence Count: {len(state.evidence)}")
        print(f"Budget Used: {MAX_RETRIEVAL_BUDGET - state.remaining_budget}")
        print(f"Stop Reason: {state.stop_reason}")

        if state.utilities:
            print(f"Final Utilities: {[round(u, 4) for u in state.utilities]}")
        if state.candidate_answers:
            print(f"Final Candidate Answers: {state.candidate_answers}")

        if state.history:
            print("-" * 80)
            print("Decision Trace:")
            for h in state.history:
                print(
                    f" step={h['step']}, action={h['action']}, "
                    f"budget={h['remaining_budget']}, "
                    f"num_evidence={h['num_evidence']}, "
                    f"best_u={max(h['utilities']) if h['utilities'] else 0.0:.4f}, "
                    f"conflict={h['conflict_uncertainty']:.4f}, "
                    f"total_u={h['total_uncertainty']:.4f}, "
                    f"delta={h.get('delta_uncertainty', 0.0):.4f}, "
                    f"gain={h.get('evidence_gain', 0.0):.4f}"
                )
        if state.evidence:
            print("-" * 80)
            print("Final Evidence Snapshot:")
            for i, ev in enumerate(state.evidence):
                util = state.utilities[i] if i < len(state.utilities) else None
                ans = (
                    state.candidate_answers[i]
                    if i < len(state.candidate_answers)
                    else None
                )
                text_preview = ev.get("text", "")[:120].replace("\n", " ")
                print(
                    f"[{i+1}] score={ev.get('score', 0.0):.4f}, "
                    f"utility={util:.4f} "
                    f"answer={ans!r} "
                    f"text={text_preview}"
                )

    metrics = {
        "accuracy": compute_accuracy(records),
        "auroc": compute_auroc(records),
        "avg_uncertainty": compute_avg_uncertainty(records),
        "selective_accuracy_80": selective_accuracy(records, keep_ratio=0.8),
        "decision_summary": summarize_decision_records(records),
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(records, OUTPUTS_DIR / "phase5_predictions.json")
    save_json(metrics, OUTPUTS_DIR / "phase5_metrics.json")

    print("\n" + "=" * 80)
    print("Phase 5 metrics")
    print(metrics)


if __name__ == "__main__":
    main()
