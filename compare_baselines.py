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
)
import argparse

from utils.io_utils import load_json, save_json
from retriever.bm25_retriever import BM25Retriever
from inference.predict_utility import UtilityPredictor
from generator.deepseek_answerer import DeepSeekAnswerer
from uncertainty.signals import DecisionAwareUncertainty
from controller.policy import RuleBasedPolicy
from decision.loop import DecisionAwareRAG

from baselines.qa_baselines import (
    apply_matched_coverage,
    run_majority_vote,
    run_single_shot,
    run_single_shot_rerank,
    run_single_shot_abstain,
    run_decision_loop,
)

from evaluation.metrics import (
    compute_accuracy,
    compute_answer_metrics,
    compute_auroc,
    compute_avg_uncertainty,
    selective_accuracy,
)
from evaluation.decision_metrics import summarize_decision_records


def evaluate_records(records):
    return {
        "accuracy": compute_accuracy(records),
        "answer_metrics": compute_answer_metrics(records),
        "auroc": compute_auroc(records),
        "avg_uncertainty": compute_avg_uncertainty(records),
        "selective_accuracy_80": selective_accuracy(records, keep_ratio=0.8),
        "decision_summary": summarize_decision_records(records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=DEFAULT_EVAL_SAMPLE_LIMIT)
    args = parser.parse_args()

    dataset = load_json(MINI_DATASET_PATH)
    if args.max_samples is not None and args.max_samples > 0:
        dataset = dataset[: args.max_samples]

    retriever = BM25Retriever(CORPUS_PATH)
    utility_predictor = UtilityPredictor()
    answerer = DeepSeekAnswerer()
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

    methods = {
        "single_shot": [],
        "single_shot_rerank": [],
        "majority_vote": [],
        "single_shot_abstain": [],
        "decision_loop": [],
    }

    for sample in dataset:
        question = sample["question"]
        gold_answers = sample["gold_answers"]

        methods["single_shot"].append(
            run_single_shot(
                question=question,
                gold_answers=gold_answers,
                retriever=retriever,
                utility_predictor=utility_predictor,
                answerer=answerer,
                uncertainty_scorer=uncertainty_scorer,
                top_k=INITIAL_TOP_K,
            )
        )

        methods["single_shot_rerank"].append(
            run_single_shot_rerank(
                question=question,
                gold_answers=gold_answers,
                retriever=retriever,
                utility_predictor=utility_predictor,
                answerer=answerer,
                uncertainty_scorer=uncertainty_scorer,
                top_k=max(INITIAL_TOP_K, 5),
                keep_top_m=INITIAL_TOP_K,
            )
        )

        methods["majority_vote"].append(
            run_majority_vote(
                question=question,
                gold_answers=gold_answers,
                retriever=retriever,
                utility_predictor=utility_predictor,
                answerer=answerer,
                uncertainty_scorer=uncertainty_scorer,
                top_k=max(INITIAL_TOP_K, 5),
            )
        )

        methods["single_shot_abstain"].append(
            run_single_shot_abstain(
                question=question,
                gold_answers=gold_answers,
                retriever=retriever,
                utility_predictor=utility_predictor,
                answerer=answerer,
                uncertainty_scorer=uncertainty_scorer,
                top_k=INITIAL_TOP_K,
                tau_answer=TAU_ANSWER,
                tau_conflict=TAU_CONFLICT,
                tau_stop=TAU_STOP,
            )
        )

        methods["decision_loop"].append(
            run_decision_loop(
                question=question,
                gold_answers=gold_answers,
                runner=runner,
            )
        )

    decision_answer_rate = evaluate_records(methods["decision_loop"])[
        "decision_summary"
    ]["answer_rate"]
    methods["single_shot_matched_coverage"] = apply_matched_coverage(
        methods["single_shot"],
        target_answer_rate=decision_answer_rate,
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for method_name, records in methods.items():
        save_json(records, OUTPUTS_DIR / f"{method_name}_predictions.json")
        metrics = evaluate_records(records)
        all_metrics[method_name] = metrics

        print("\n" + "=" * 80)
        print(f"Method: {method_name}")
        print(metrics)

    save_json(all_metrics, OUTPUTS_DIR / "phase5_compare_metrics.json")
    print("\nSaved comparison metrics to outputs/phase5_compare_metrics.json")


if __name__ == "__main__":
    main()
