from copy import deepcopy
import argparse

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


def evaluate_records(records):
    return {
        "accuracy": compute_accuracy(records),
        "auroc": compute_auroc(records),
        "avg_uncertainty": compute_avg_uncertainty(records),
        "selective_accuracy_80": selective_accuracy(records, keep_ratio=0.8),
        "decision_summary": summarize_decision_records(records),
    }


def run_one_setting(
    setting_name,
    boundary_cfg,
    dataset,
    retriever,
    utility_predictor,
    answerer,
    uncertainty_scorer,
):
    policy = RuleBasedPolicy(
        tau_answer=TAU_ANSWER,
        tau_retrieve=TAU_RETRIEVE,
        tau_conflict=TAU_CONFLICT,
        tau_stop=TAU_STOP,
        tau_delta=TAU_DELTA,
        tau_gain=TAU_GAIN,
        answer_min_utility=boundary_cfg["answer_min_utility"],
        answer_max_conflict=boundary_cfg["answer_max_conflict"],
        answer_max_total_uncertainty=boundary_cfg["answer_max_total_uncertainty"],
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
                "uncertainty": state.total_uncertainty,
                "retrieval_uncertainty": state.retrieval_uncertainty,
                "conflict_uncertainty": state.conflict_uncertainty,
                "stability_uncertainty": state.stability_uncertainty,
                "steps": state.step,
                "num_evidence": len(state.evidence),
                "budget_used": MAX_RETRIEVAL_BUDGET - state.remaining_budget,
                "stop_reason": state.stop_reason,
                "history": deepcopy(state.history),
            }
        )

    metrics = evaluate_records(records)
    return records, metrics


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

    sweep_settings = {
        "D_loose_1": {
            "answer_min_utility": 0.06,
            "answer_max_conflict": 0.90,
            "answer_max_total_uncertainty": 0.95,
        },
        "E_loose_2": {
            "answer_min_utility": 0.08,
            "answer_max_conflict": 0.88,
            "answer_max_total_uncertainty": 0.92,
        },
        "F_balanced": {
            "answer_min_utility": 0.10,
            "answer_max_conflict": 0.85,
            "answer_max_total_uncertainty": 0.90,
        },
        "G_selective": {
            "answer_min_utility": 0.12,
            "answer_max_conflict": 0.82,
            "answer_max_total_uncertainty": 0.88,
        },
        "H_more_selective": {
            "answer_min_utility": 0.14,
            "answer_max_conflict": 0.80,
            "answer_max_total_uncertainty": 0.86,
        },
    }

    all_results = {}

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    for setting_name, boundary_cfg in sweep_settings.items():
        print("\n" + "=" * 80)
        print(f"Running setting: {setting_name}")
        print(boundary_cfg)

        records, metrics = run_one_setting(
            setting_name=setting_name,
            boundary_cfg=boundary_cfg,
            dataset=dataset,
            retriever=retriever,
            utility_predictor=utility_predictor,
            answerer=answerer,
            uncertainty_scorer=uncertainty_scorer,
        )

        all_results[setting_name] = {
            "boundary": boundary_cfg,
            "metrics": metrics,
        }

        save_json(records, OUTPUTS_DIR / f"phase5_{setting_name}_predictions.json")

        print(metrics)

    save_json(all_results, OUTPUTS_DIR / "phase5_sweep_results.json")
    print("\nSaved sweep results to outputs/phase5_sweep_results.json")


if __name__ == "__main__":
    main()
