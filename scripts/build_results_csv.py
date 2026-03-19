import json
import os
import re
import pandas as pd

OUTPUT_DIR = "outputs"
SAVE_NAME = "final_results.csv"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_policy_from_filename(fname: str):
    """
    从文件名中解析 method / policy
    """
    base = fname.replace("_predictions.json", "")

    if base == "single_shot":
        return "single_shot", "none"
    if base == "single_shot_rerank":
        return "single_shot_rerank", "none"
    if base == "single_shot_abstain":
        return "single_shot_abstain", "none"
    if base == "decision_loop":
        return "decision_loop", "default"
    if base == "phase5":
        return "decision_loop", "phase5_default"

    m = re.match(r"phase5_([A-Z])_(.+)", base)
    if m:
        _, policy_name = m.groups()
        return "decision_loop", policy_name

    return "unknown", base


def get_final_action(item: dict):
    return str(item.get("final_action", "")).strip().upper()


def infer_answered(item: dict):
    final_action = get_final_action(item)
    if final_action == "ANSWER":
        return 1
    if final_action in {"ABSTAIN", "PARTIAL"}:
        return 0

    final_answer = item.get("final_answer", None)
    if final_answer is None:
        return 0
    if str(final_answer).strip() == "":
        return 0
    return 1


def extract_last_history_fields(item: dict):
    """
    从 history 最后一步提取更细的运行时字段
    """
    history = item.get("history", [])
    if isinstance(history, list) and len(history) > 0 and isinstance(history[-1], dict):
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


def extract_row(item, fname, idx, method, policy):
    final_action = get_final_action(item)
    answered = infer_answered(item)

    row = {
        "source_file": fname,
        "row_id": idx,
        "qid": idx,  # 你的样例里没有 question_id，就先用行号
        "method": method,
        "policy": policy,

        "question": item.get("question"),
        "gold_answers": json.dumps(item.get("gold_answers", []), ensure_ascii=False),
        "final_answer": item.get("final_answer"),
        "final_action": final_action,

        "correct": item.get("correct", 0),
        "answered": answered,

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


def main():
    if not os.path.exists(OUTPUT_DIR):
        raise FileNotFoundError(f"Output directory not found: {OUTPUT_DIR}")

    files = sorted([
        f for f in os.listdir(OUTPUT_DIR)
        if f.endswith("_predictions.json")
    ])

    if not files:
        raise FileNotFoundError(f"No *_predictions.json files found in {OUTPUT_DIR}")

    rows = []

    for fname in files:
        path = os.path.join(OUTPUT_DIR, fname)
        method, policy = normalize_policy_from_filename(fname)

        try:
            data = load_json(path)
        except Exception as e:
            print(f"[ERROR] Failed to read {fname}: {e}")
            continue

        if not isinstance(data, list):
            print(f"[WARN] Skip {fname}: JSON root is not a list")
            continue

        loaded = 0
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                print(f"[WARN] Skip row {idx} in {fname}: item is not dict")
                continue
            rows.append(extract_row(item, fname, idx, method, policy))
            loaded += 1

        print(f"[OK] Loaded {fname} -> {loaded} rows | method={method}, policy={policy}")

    df = pd.DataFrame(rows)

    numeric_cols = [
        "correct", "answered",
        "uncertainty", "retrieval_uncertainty", "conflict_uncertainty", "stability_uncertainty",
        "steps", "num_evidence", "budget_used",
        "history_len", "last_history_step", "remaining_budget",
        "best_answer_weight", "best_utility",
        "delta_uncertainty", "evidence_gain",
        "answer_utility", "continue_utility", "service_utility",
        "generation_entropy", "utility_uncertainty", "stability_score",
        "history_total_uncertainty",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    save_path = os.path.join(OUTPUT_DIR, SAVE_NAME)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print(f"Saved unified CSV: {save_path}")
    print(f"Total rows: {len(df)}")
    print("==============================\n")

    summary = (
        df.groupby(["method", "policy"], dropna=False)
        .agg(
            count=("qid", "count"),
            accuracy=("correct", "mean"),
            answer_rate=("answered", "mean"),
            avg_uncertainty=("uncertainty", "mean"),
            avg_retrieval_uncertainty=("retrieval_uncertainty", "mean"),
            avg_conflict_uncertainty=("conflict_uncertainty", "mean"),
            avg_stability_uncertainty=("stability_uncertainty", "mean"),
            avg_steps=("steps", "mean"),
            avg_evidence=("num_evidence", "mean"),
            avg_budget=("budget_used", "mean"),
        )
        .reset_index()
        .sort_values(["method", "policy"])
    )

    summary_path = os.path.join(OUTPUT_DIR, "final_results_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"Saved summary CSV: {summary_path}")
    print("\nPreview:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()