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

from baselines.phase5_baselines import (
    run_single_shot,
    run_single_shot_rerank,
    run_single_shot_abstain,
    run_decision_loop,
)

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


def main():
    dataset = load_json(MINI_DATASET_PATH)
    dataset = dataset[:10]

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

    methods = {
        "single_shot": [],
        "single_shot_rerank": [],
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