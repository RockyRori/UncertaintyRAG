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
    ANSWER_MIN_UTILITY,
    ANSWER_MIN_BEST_UTILITY,
    ANSWER_MAX_CONFLICT,
    ANSWER_MAX_TOTAL_UNCERTAINTY,
    ANSWER_MIN_STABILITY,
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
    compute_answer_metrics,
    compute_auroc,
    compute_avg_uncertainty,
    selective_accuracy,
)
from evaluation.decision_metrics import summarize_decision_records
from utils.text_utils import qa_metrics


def evaluate_records(records):
    return {
        "accuracy": compute_accuracy(records),
        "answer_metrics": compute_answer_metrics(records),
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
        answer_min_utility=boundary_cfg.get("answer_min_utility", ANSWER_MIN_UTILITY),
        answer_min_best_utility=boundary_cfg.get(
            "answer_min_best_utility", ANSWER_MIN_BEST_UTILITY
        ),
        answer_max_conflict=boundary_cfg.get("answer_max_conflict", ANSWER_MAX_CONFLICT),
        answer_max_total_uncertainty=boundary_cfg.get(
            "answer_max_total_uncertainty", ANSWER_MAX_TOTAL_UNCERTAINTY
        ),
        tau_stability=boundary_cfg.get("tau_stability", ANSWER_MIN_STABILITY),
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
            "answer_min_utility": 0.30,
            "answer_min_best_utility": 0.20,
            "answer_max_conflict": 0.75,
            "answer_max_total_uncertainty": 0.90,
            "tau_stability": 0.02,
        },
        "E_loose_2": {
            "answer_min_utility": 0.40,
            "answer_min_best_utility": 0.28,
            "answer_max_conflict": 0.65,
            "answer_max_total_uncertainty": 0.80,
            "tau_stability": 0.08,
        },
        "F_balanced": {
            "answer_min_utility": ANSWER_MIN_UTILITY,
            "answer_min_best_utility": ANSWER_MIN_BEST_UTILITY,
            "answer_max_conflict": ANSWER_MAX_CONFLICT,
            "answer_max_total_uncertainty": ANSWER_MAX_TOTAL_UNCERTAINTY,
            "tau_stability": ANSWER_MIN_STABILITY,
        },
        "G_selective": {
            "answer_min_utility": 0.60,
            "answer_min_best_utility": 0.50,
            "answer_max_conflict": 0.45,
            "answer_max_total_uncertainty": 0.60,
            "tau_stability": 0.22,
        },
        "H_more_selective": {
            "answer_min_utility": 0.72,
            "answer_min_best_utility": 0.65,
            "answer_max_conflict": 0.35,
            "answer_max_total_uncertainty": 0.50,
            "tau_stability": 0.35,
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
