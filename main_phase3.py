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
from evaluation.metrics import compute_accuracy, compute_auroc, compute_avg_uncertainty, selective_accuracy
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

    for sample in dataset[:20]:
        question = sample["question"]
        gold_answers = sample["gold_answers"]

        state = runner.run_one(question=question, gold_answers=gold_answers)

        records.append({
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
            "history": state.history,
        })

        print("=" * 80)
        print(f"Question: {question}")
        print(f"Gold: {gold_answers}")
        print(f"Action: {state.final_action}")
        print(f"Pred: {state.final_answer}")
        print(f"Correct: {state.correct}")
        print(f"Uncertainty: {state.total_uncertainty:.4f}")
        print(f"Steps: {state.step}")
        print(f"Evidence Count: {len(state.evidence)}")

    metrics = {
        "accuracy": compute_accuracy(records),
        "auroc": compute_auroc(records),
        "avg_uncertainty": compute_avg_uncertainty(records),
        "selective_accuracy_80": selective_accuracy(records, keep_ratio=0.8),
        "decision_summary": summarize_decision_records(records),
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(records, OUTPUTS_DIR / "phase3_predictions.json")
    save_json(metrics, OUTPUTS_DIR / "phase3_metrics.json")

    print("\n" + "=" * 80)
    print("Phase 3 metrics")
    print(metrics)


if __name__ == "__main__":
    main()