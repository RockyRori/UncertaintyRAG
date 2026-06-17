import numpy as np
from sklearn.metrics import roc_auc_score

from utils.text_utils import qa_metrics


def compute_accuracy(records: list[dict]) -> float:
    if not records:
        return 0.0
    return float(np.mean([r["correct"] for r in records]))


def _record_metric(record: dict, key: str) -> float:
    if key in record and record[key] is not None:
        return float(record[key])
    metrics = qa_metrics(record.get("final_answer", ""), record.get("gold_answers", []))
    return float(metrics.get(key, 0.0))


def compute_answer_metrics(records: list[dict]) -> dict:
    if not records:
        return {
            "exact_match": 0.0,
            "relaxed_match": 0.0,
            "contains_answer": 0.0,
            "token_f1": 0.0,
            "answered_exact_match": 0.0,
            "answered_relaxed_match": 0.0,
            "answered_contains_answer": 0.0,
            "answered_token_f1": 0.0,
        }

    answered = [r for r in records if r.get("final_action") == "ANSWER"]

    def mean_metric(rows: list[dict], key: str) -> float:
        if not rows:
            return 0.0
        return float(np.mean([_record_metric(r, key) for r in rows]))

    return {
        "exact_match": mean_metric(records, "exact_match"),
        "relaxed_match": mean_metric(records, "relaxed_match"),
        "contains_answer": mean_metric(records, "contains_answer"),
        "token_f1": mean_metric(records, "token_f1"),
        "answered_exact_match": mean_metric(answered, "exact_match"),
        "answered_relaxed_match": mean_metric(answered, "relaxed_match"),
        "answered_contains_answer": mean_metric(answered, "contains_answer"),
        "answered_token_f1": mean_metric(answered, "token_f1"),
    }


def compute_auroc(records: list[dict]) -> float | None:
    if not records:
        return None

    y_true = [1 - r["correct"] for r in records]   # incorrect = positive
    y_score = [r["uncertainty"] for r in records]

    if len(set(y_true)) < 2:
        return None

    return float(roc_auc_score(y_true, y_score))


def compute_avg_uncertainty(records: list[dict]):
    if not records:
        return {
            "overall": 0.0,
            "correct_only": 0.0,
            "incorrect_only": 0.0
        }

    all_u = [r["uncertainty"] for r in records]
    correct_u = [r["uncertainty"] for r in records if r["correct"] == 1]
    incorrect_u = [r["uncertainty"] for r in records if r["correct"] == 0]

    return {
        "overall": float(np.mean(all_u)) if all_u else 0.0,
        "correct_only": float(np.mean(correct_u)) if correct_u else 0.0,
        "incorrect_only": float(np.mean(incorrect_u)) if incorrect_u else 0.0,
    }


def selective_accuracy(records: list[dict], keep_ratio: float = 0.8):
    if not records:
        return {
            "kept_count": 0,
            "kept_ratio": keep_ratio,
            "accuracy": 0.0
        }

    sorted_records = sorted(records, key=lambda x: x["uncertainty"])
    keep_n = max(1, int(len(sorted_records) * keep_ratio))
    kept = sorted_records[:keep_n]

    return {
        "kept_count": keep_n,
        "kept_ratio": keep_ratio,
        "accuracy": float(np.mean([r["correct"] for r in kept]))
    }
