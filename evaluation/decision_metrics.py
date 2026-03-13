import numpy as np


def summarize_decision_records(records: list[dict]) -> dict:
    if not records:
        return {
            "count": 0,
            "answer_rate": 0.0,
            "abstain_rate": 0.0,
            "accuracy_all": 0.0,
            "accuracy_answered_only": 0.0,
            "avg_steps": 0.0,
            "avg_evidence": 0.0,
            "avg_budget_used": 0.0,
            "overconfident_error_rate": 0.0,
        }

    answered = [r for r in records if r["final_action"] == "ANSWER"]
    abstained = [r for r in records if r["final_action"] == "ABSTAIN"]

    overconfident_errors = [
        r for r in records
        if r["final_action"] == "ANSWER" and r["correct"] == 0 and r["uncertainty"] < 0.30
    ]

    return {
        "count": len(records),
        "answer_rate": len(answered) / len(records),
        "abstain_rate": len(abstained) / len(records),
        "accuracy_all": float(np.mean([r["correct"] for r in records])),
        "accuracy_answered_only": float(np.mean([r["correct"] for r in answered])) if answered else 0.0,
        "avg_steps": float(np.mean([r["steps"] for r in records])),
        "avg_evidence": float(np.mean([r["num_evidence"] for r in records])),
        "avg_budget_used": float(np.mean([r.get("budget_used", 0) for r in records])),
        "overconfident_error_rate": len(overconfident_errors) / len(records),
    }