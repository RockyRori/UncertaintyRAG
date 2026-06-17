import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import OUTPUTS_DIR
from utils.text_utils import qa_metrics

OUTPUT_DIR = str(OUTPUTS_DIR)
SAVE_NAME = "final_results_v2.csv"
SUMMARY_NAME = "final_results_summary_v2.csv"
ACTION_NAME = "final_action_history_v2.csv"
FIVE_DATASET_COMBINED = "five_datasets/all_dataset_predictions_v2.csv"
FIVE_DATASET_SUMMARY = "five_datasets/all_dataset_metrics_v2.csv"

LEGACY_PREDICTION_FILES = {
    "phase3_predictions.json",
    "phase5_A_aggressive_predictions.json",
    "phase5_B_medium_predictions.json",
    "phase5_C_conservative_predictions.json",
}


def reset_output_file(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


COMPARE_BASELINES = {
    "single_shot": ("single_shot", "none", "compare_small"),
    "single_shot_rerank": ("single_shot_rerank", "none", "compare_small"),
    "majority_vote": ("majority_vote", "none", "compare_small"),
    "single_shot_abstain": ("single_shot_abstain", "none", "compare_small"),
    "single_shot_matched_coverage": (
        "single_shot_matched_coverage",
        "matched_coverage",
        "compare_small",
    ),
    "decision_loop": ("decision_loop", "default", "compare_small"),
    "phase5": ("decision_loop", "phase5_default", "compare_small"),
}


POLICY_DISPLAY = {
    "aggressive": "Aggressive",
    "medium": "Medium",
    "conservative": "Conservative",
    "loose_1": "Loose-1",
    "loose_2": "Loose-2",
    "balanced": "Balanced",
    "selective": "Selective",
    "more_selective": "More Selective",
    "default": "Default",
    "phase5_default": "Phase5 Default",
    "none": "No Policy",
    "matched_coverage": "Matched Coverage",
}


def normalize_policy_from_filename(fname: str) -> Tuple[str, str, str]:
    base = fname.replace("_predictions.json", "")
    if base in COMPARE_BASELINES:
        return COMPARE_BASELINES[base]
    m = re.match(r"phase5_([A-Z])_(.+)", base)
    if m:
        _, policy_name = m.groups()
        return "decision_loop", policy_name, "policy_sweep"

    return "unknown", base, "other"


def get_final_action(item: Dict[str, Any]) -> str:
    return str(item.get("final_action", "")).strip().upper()


def infer_answered(item: Dict[str, Any]) -> int:
    final_action = get_final_action(item)
    if final_action == "ANSWER":
        return 1
    if final_action in {"ABSTAIN", "PARTIAL"}:
        return 0

    final_answer = item.get("final_answer")
    if final_answer is None:
        return 0
    if str(final_answer).strip() == "":
        return 0
    if str(final_answer).strip().upper() == "ABSTAIN":
        return 0
    return 1


def extract_last_history_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    history = item.get("history", [])
    if isinstance(history, list) and history and isinstance(history[-1], dict):
        h = history[-1]
        return {
            "last_history_step": h.get("step"),
            "last_history_action": h.get("action"),
            "remaining_budget": h.get("remaining_budget"),
            "best_answer": h.get("best_answer"),
            "best_answer_weight": h.get("best_answer_weight"),
            "best_utility": h.get("best_utility"),
            "delta_uncertainty": h.get("delta_uncertainty"),
            "evidence_gain": h.get("evidence_gain"),
            "answer_utility": h.get("answer_utility"),
            "continue_utility": h.get("continue_utility"),
            "service_utility": h.get("service_utility"),
            "generation_entropy": h.get("generation_entropy"),
            "utility_uncertainty": h.get("utility_uncertainty"),
            "stability_score": h.get("stability_score"),
            "history_total_uncertainty": h.get("total_uncertainty"),
        }
    return {
        "last_history_step": None,
        "last_history_action": None,
        "remaining_budget": None,
        "best_answer": None,
        "best_answer_weight": None,
        "best_utility": None,
        "delta_uncertainty": None,
        "evidence_gain": None,
        "answer_utility": None,
        "continue_utility": None,
        "service_utility": None,
        "generation_entropy": None,
        "utility_uncertainty": None,
        "stability_score": None,
        "history_total_uncertainty": None,
    }


def extract_action_rows(item: Dict[str, Any], fname: str, idx: int, method: str, policy: str, experiment_group: str) -> List[Dict[str, Any]]:
    rows = []
    for h_idx, h in enumerate(item.get("history", []) or [], start=1):
        if not isinstance(h, dict):
            continue
        rows.append({
            "source_file": fname,
            "row_id": idx,
            "qid": idx,
            "method": method,
            "policy": policy,
            "policy_display": POLICY_DISPLAY.get(policy, policy),
            "experiment_group": experiment_group,
            "question": item.get("question"),
            "final_action": get_final_action(item),
            "correct": item.get("correct", 0),
            "answered": infer_answered(item),
            "history_index": h_idx,
            "step": h.get("step"),
            "action": h.get("action"),
            "last_action": h.get("last_action"),
            "remaining_budget": h.get("remaining_budget"),
            "num_evidence": h.get("num_evidence"),
            "best_answer": h.get("best_answer"),
            "best_answer_weight": h.get("best_answer_weight"),
            "best_utility": h.get("best_utility"),
            "delta_uncertainty": h.get("delta_uncertainty"),
            "evidence_gain": h.get("evidence_gain"),
            "answer_utility": h.get("answer_utility"),
            "continue_utility": h.get("continue_utility"),
            "service_utility": h.get("service_utility"),
            "generation_entropy": h.get("generation_entropy"),
            "utility_uncertainty": h.get("utility_uncertainty"),
            "stability_score": h.get("stability_score"),
            "total_uncertainty": h.get("total_uncertainty"),
            "retrieval_uncertainty": h.get("retrieval_uncertainty"),
            "conflict_uncertainty": h.get("conflict_uncertainty"),
            "stability_uncertainty": h.get("stability_uncertainty"),
        })
    return rows


def extract_prediction_row(item: Dict[str, Any], fname: str, idx: int, method: str, policy: str, experiment_group: str) -> Dict[str, Any]:
    final_action = get_final_action(item)
    answered = infer_answered(item)
    if final_action == "ANSWER":
        metric_values = qa_metrics(
            item.get("final_answer", ""),
            item.get("gold_answers", []),
        )
    else:
        metric_values = {
            "exact_match": 0.0,
            "relaxed_match": 0.0,
            "contains_answer": 0.0,
            "token_f1": 0.0,
        }
    row = {
        "source_file": fname,
        "row_id": idx,
        "qid": idx,
        "method": method,
        "policy": policy,
        "policy_display": POLICY_DISPLAY.get(policy, policy),
        "experiment_group": experiment_group,
        "question": item.get("question"),
        "gold_answers": json.dumps(item.get("gold_answers", []), ensure_ascii=False),
        "final_answer": item.get("final_answer"),
        "final_action": final_action,
        "correct": item.get("correct", 0),
        "exact_match": item.get("exact_match", metric_values["exact_match"]),
        "relaxed_match": item.get("relaxed_match", metric_values["relaxed_match"]),
        "contains_answer": item.get("contains_answer", metric_values["contains_answer"]),
        "token_f1": item.get("token_f1", metric_values["token_f1"]),
        "answered": answered,
        "answered_correct": item.get("correct", 0) if answered else None,
        "answered_relaxed_match": item.get("relaxed_match", metric_values["relaxed_match"]) if answered else None,
        "answered_token_f1": item.get("token_f1", metric_values["token_f1"]) if answered else None,
        "answered_contains_answer": item.get("contains_answer", metric_values["contains_answer"]) if answered else None,
        "uncertainty": item.get("uncertainty"),
        "retrieval_uncertainty": item.get("retrieval_uncertainty"),
        "conflict_uncertainty": item.get("conflict_uncertainty"),
        "stability_uncertainty": item.get("stability_uncertainty"),
        "steps": item.get("steps"),
        "num_evidence": item.get("num_evidence"),
        "budget_used": item.get("budget_used"),
        "stop_reason": item.get("stop_reason"),
        "history_len": len(item.get("history", [])) if isinstance(item.get("history", []), list) else 0,
    }
    row.update(extract_last_history_fields(item))
    return row


NUMERIC_COLS = [
    "correct", "exact_match", "relaxed_match", "contains_answer", "token_f1",
    "answered", "answered_correct", "answered_relaxed_match", "answered_token_f1", "answered_contains_answer",
    "uncertainty", "retrieval_uncertainty", "conflict_uncertainty", "stability_uncertainty",
    "steps", "num_evidence", "budget_used",
    "history_len", "last_history_step", "remaining_budget",
    "best_answer_weight", "best_utility",
    "delta_uncertainty", "evidence_gain",
    "answer_utility", "continue_utility", "service_utility",
    "generation_entropy", "utility_uncertainty", "stability_score",
    "history_total_uncertainty",
]


ACTION_NUMERIC_COLS = [
    "correct", "answered", "history_index", "step", "remaining_budget", "num_evidence",
    "best_answer_weight", "best_utility", "delta_uncertainty", "evidence_gain",
    "answer_utility", "continue_utility", "service_utility", "generation_entropy",
    "utility_uncertainty", "stability_score", "total_uncertainty", "retrieval_uncertainty",
    "conflict_uncertainty", "stability_uncertainty",
]


def convert_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_prediction_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted([
        f for f in os.listdir(OUTPUT_DIR)
        if f.endswith("_predictions.json") and f not in LEGACY_PREDICTION_FILES
    ])
    if not files:
        raise FileNotFoundError(f"No *_predictions.json files found in {OUTPUT_DIR}")

    pred_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []

    for fname in files:
        path = os.path.join(OUTPUT_DIR, fname)
        method, policy, experiment_group = normalize_policy_from_filename(fname)
        data = load_json(path)
        if not isinstance(data, list):
            continue

        loaded = 0
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            pred_rows.append(extract_prediction_row(item, fname, idx, method, policy, experiment_group))
            action_rows.extend(extract_action_rows(item, fname, idx, method, policy, experiment_group))
            loaded += 1
        print(f"[OK] Loaded {fname:<40} -> {loaded:>3} rows | method={method}, policy={policy}, group={experiment_group}")

    pred_df = convert_numeric(pd.DataFrame(pred_rows), NUMERIC_COLS)
    action_df = convert_numeric(pd.DataFrame(action_rows), ACTION_NUMERIC_COLS)
    return pred_df, action_df


def build_prediction_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    def answered_only_acc(x: pd.Series) -> Optional[float]:
        valid = x.dropna()
        return float(valid.mean()) if len(valid) else None

    summary = (
        pred_df.groupby(["experiment_group", "method", "policy", "policy_display"], dropna=False)
        .agg(
            count=("qid", "count"),
            accuracy=("correct", "mean"),
            answer_rate=("answered", "mean"),
            answered_only_accuracy=("answered_correct", answered_only_acc),
            answered_only_relaxed_match=("answered_relaxed_match", answered_only_acc),
            answered_only_token_f1=("answered_token_f1", answered_only_acc),
            answered_only_contains=("answered_contains_answer", answered_only_acc),
            relaxed_match=("relaxed_match", "mean"),
            token_f1=("token_f1", "mean"),
            contains_answer=("contains_answer", "mean"),
            avg_uncertainty=("uncertainty", "mean"),
            avg_retrieval_uncertainty=("retrieval_uncertainty", "mean"),
            avg_conflict_uncertainty=("conflict_uncertainty", "mean"),
            avg_stability_uncertainty=("stability_uncertainty", "mean"),
            avg_steps=("steps", "mean"),
            avg_evidence=("num_evidence", "mean"),
            avg_budget=("budget_used", "mean"),
        )
        .reset_index()
        .sort_values(["experiment_group", "method", "policy"])
    )
    return summary


def build_five_dataset_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_dir = os.path.join(OUTPUT_DIR, "five_datasets")
    if not os.path.isdir(base_dir):
        return pd.DataFrame(), pd.DataFrame()

    pred_frames = []
    metric_rows = []

    for dataset in sorted(os.listdir(base_dir)):
        ds_dir = os.path.join(base_dir, dataset)
        if not os.path.isdir(ds_dir):
            continue

        pred_path = os.path.join(ds_dir, "test_predictions.csv")
        metrics_path = os.path.join(ds_dir, "metrics.json")
        if os.path.exists(pred_path):
            df = pd.read_csv(pred_path)
            df["dataset"] = dataset
            pred_frames.append(df)
        if os.path.exists(metrics_path):
            metrics = load_json(metrics_path)
            metrics_row = {
                "dataset": dataset,
                "train_size": metrics.get("train_size"),
                "test_size": metrics.get("test_size"),
                "val_accuracy": metrics.get("val_accuracy"),
                "val_precision": metrics.get("val_precision"),
                "val_recall": metrics.get("val_recall"),
                "val_f1": metrics.get("val_f1"),
                "val_auroc": metrics.get("val_auroc"),
                "best_threshold": metrics.get("best_threshold"),
                "best_epoch": metrics.get("best_epoch"),
                "accuracy": metrics.get("accuracy"),
                "exact_match": metrics.get("answer_metrics", {}).get("exact_match", metrics.get("accuracy")),
                "relaxed_match": metrics.get("answer_metrics", {}).get("relaxed_match"),
                "contains_answer": metrics.get("answer_metrics", {}).get("contains_answer"),
                "token_f1": metrics.get("answer_metrics", {}).get("token_f1"),
                "answered_exact_match": metrics.get("answer_metrics", {}).get("answered_exact_match"),
                "answered_relaxed_match": metrics.get("answer_metrics", {}).get("answered_relaxed_match"),
                "answered_contains_answer": metrics.get("answer_metrics", {}).get("answered_contains_answer"),
                "answered_token_f1": metrics.get("answer_metrics", {}).get("answered_token_f1"),
                "auroc": metrics.get("auroc"),
                "corpus_placeholder_ratio": metrics.get("corpus_quality", {}).get("placeholder_ratio"),
                "corpus_path": metrics.get("corpus_path"),
                "avg_uncertainty_overall": metrics.get("avg_uncertainty", {}).get("overall"),
                "avg_uncertainty_correct": metrics.get("avg_uncertainty", {}).get("correct_only"),
                "avg_uncertainty_incorrect": metrics.get("avg_uncertainty", {}).get("incorrect_only"),
                "selective_accuracy_80": metrics.get("selective_accuracy_80", {}).get("accuracy"),
                "kept_count_80": metrics.get("selective_accuracy_80", {}).get("kept_count"),
                "kept_ratio_80": metrics.get("selective_accuracy_80", {}).get("kept_ratio"),
            }
            metric_rows.append(metrics_row)

    pred_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    metric_df = pd.DataFrame(metric_rows)
    pred_df = convert_numeric(pred_df, [
        "correct", "exact_match", "relaxed_match", "contains_answer", "token_f1",
        "uncertainty", "retrieval_uncertainty", "conflict_uncertainty",
        "stability_uncertainty", "steps", "num_evidence", "budget_used"
    ])
    if not pred_df.empty:
        pred_df["error_label"] = 1 - pred_df["correct"]
    return pred_df, metric_df


def main():
    if not os.path.exists(OUTPUT_DIR):
        raise FileNotFoundError(f"Output directory not found: {OUTPUT_DIR}")

    pred_df, action_df = build_prediction_tables()
    pred_summary = build_prediction_summary(pred_df)
    pred_path = os.path.join(OUTPUT_DIR, SAVE_NAME)
    summary_path = os.path.join(OUTPUT_DIR, SUMMARY_NAME)
    action_path = os.path.join(OUTPUT_DIR, ACTION_NAME)
    for output_path in [pred_path, summary_path, action_path]:
        reset_output_file(output_path)

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    pred_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    action_df.to_csv(action_path, index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print(" -", pred_path)
    print(" -", summary_path)
    print(" -", action_path)

    fd_pred, fd_metrics = build_five_dataset_tables()
    if not fd_pred.empty:
        combined_path = os.path.join(OUTPUT_DIR, FIVE_DATASET_COMBINED)
        reset_output_file(combined_path)
        fd_pred.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(" -", combined_path)
    if not fd_metrics.empty:
        five_summary_path = os.path.join(OUTPUT_DIR, FIVE_DATASET_SUMMARY)
        reset_output_file(five_summary_path)
        fd_metrics.to_csv(five_summary_path, index=False, encoding="utf-8-sig")
        print(" -", five_summary_path)

    print("\nSummary preview:")
    print(pred_summary.to_string(index=False))

    if not action_df.empty:
        print("\nAction distribution preview:")
        act = action_df.groupby(["experiment_group", "policy", "action"]).size().reset_index(name="count")
        print(act.to_string(index=False))


if __name__ == "__main__":
    main()
