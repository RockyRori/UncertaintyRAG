from config import (
    CORPUS_PATH,
    DEFAULT_EVAL_SAMPLE_LIMIT,
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
    ANSWER_MIN_UTILITY,
    ANSWER_MIN_BEST_UTILITY,
    ANSWER_MAX_CONFLICT,
    ANSWER_MAX_TOTAL_UNCERTAINTY,
    ANSWER_MIN_STABILITY,
    ANSWER_HIGH_UTILITY,
    ANSWER_HIGH_UTILITY_MIN_ANSWER_UTILITY,
    ANSWER_HIGH_UTILITY_MAX_TOTAL_UNCERTAINTY,
    UNCERTAINTY_ALPHA,
    UNCERTAINTY_BETA,
    UNCERTAINTY_GAMMA,
    GENERATOR_MODEL_NAME,
    MAX_INPUT_LENGTH,
    MAX_NEW_TOKENS,
)
import argparse
from utils.io_utils import load_json, save_json
from retriever.bm25_retriever import BM25Retriever
from inference.predict_utility import UtilityPredictor
from generator.simple_answerer import SimpleAnswerer
from uncertainty.signals import DecisionAwareUncertainty
from controller.policy import RuleBasedPolicy
from decision.loop import DecisionAwareRAG
from evaluation.metrics import (
    compute_accuracy,
    compute_answer_metrics,
    compute_auroc,
    compute_avg_uncertainty,
    selective_accuracy,
)
from evaluation.decision_metrics import summarize_decision_records
from utils.text_utils import qa_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=DEFAULT_EVAL_SAMPLE_LIMIT)
    args = parser.parse_args()

    dataset = load_json(MINI_DATASET_PATH)
    if args.max_samples is not None and args.max_samples > 0:
        dataset = dataset[: args.max_samples]

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
        answer_min_utility=ANSWER_MIN_UTILITY,
        answer_min_best_utility=ANSWER_MIN_BEST_UTILITY,
        answer_max_conflict=ANSWER_MAX_CONFLICT,
        answer_max_total_uncertainty=ANSWER_MAX_TOTAL_UNCERTAINTY,
        tau_stability=ANSWER_MIN_STABILITY,
        answer_high_utility=ANSWER_HIGH_UTILITY,
        answer_high_utility_min_answer_utility=ANSWER_HIGH_UTILITY_MIN_ANSWER_UTILITY,
        answer_high_utility_max_total_uncertainty=ANSWER_HIGH_UTILITY_MAX_TOTAL_UNCERTAINTY,
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

    for sample in dataset:
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
                **(
                    qa_metrics(state.final_answer, gold_answers)
                    if state.final_action == "ANSWER"
                    else {
                        "exact_match": 0.0,
                        "contains_answer": 0.0,
                        "token_f1": 0.0,
                    }
                ),

                "uncertainty": state.total_uncertainty,
                "generation_entropy": state.generation_entropy,
                "utility_uncertainty": state.utility_uncertainty,
                "stability_score": state.stability_score,

                "retrieval_uncertainty": state.retrieval_uncertainty,
                "conflict_uncertainty": state.conflict_uncertainty,
                "stability_uncertainty": state.stability_uncertainty,

                "best_answer": state.best_answer,
                "best_answer_weight": state.best_answer_weight,
                "best_utility": state.best_utility,

                "answer_utility": state.answer_utility,
                "continue_utility": state.continue_utility,
                "service_utility": state.service_utility,

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

        print(f"Generation Entropy : {state.generation_entropy:.4f}")
        print(f"Utility Uncertainty: {state.utility_uncertainty:.4f}")
        print(f"Stability Score    : {state.stability_score:.4f}")
        print(f"Total Uncertainty  : {state.total_uncertainty:.4f}")

        print(f"Answer Utility  : {state.answer_utility:.4f}")
        print(f"Continue Utility: {state.continue_utility:.4f}")
        print(f"Service Utility : {state.service_utility:.4f}")

        print(f"Best Answer      : {state.best_answer}")
        print(f"Best Answer Weight: {state.best_answer_weight:.4f}")
        print(f"Best Utility     : {state.best_utility:.4f}")

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
                    f"best_answer={h['best_answer']!r}, "
                    f"best_u={h['best_utility']:.4f}, "
                    f"H={h['generation_entropy']:.4f}, "
                    f"utilU={h['utility_uncertainty']:.4f}, "
                    f"S={h['stability_score']:.4f}, "
                    f"total_u={h['total_uncertainty']:.4f}, "
                    f"ans_util={h['answer_utility']:.4f}, "
                    f"cont_util={h['continue_utility']:.4f}, "
                    f"delta={h.get('delta_uncertainty', 0.0):.4f}, "
                    f"gain={h.get('evidence_gain', 0.0):.4f}"
                )

    metrics = {
        "accuracy": compute_accuracy(records),
        "answer_metrics": compute_answer_metrics(records),
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
