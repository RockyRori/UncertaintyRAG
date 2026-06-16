import json
import os
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd

from config import OUTPUTS_DIR

OUTPUT_DIR = str(OUTPUTS_DIR)
TABLE_DIR = os.path.join(OUTPUT_DIR, "paper_tables")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "paper_figure_data")

PRED_PATH = os.path.join(OUTPUT_DIR, "final_results_v2.csv")
ACTION_PATH = os.path.join(OUTPUT_DIR, "final_action_history_v2.csv")
DATASET_PRED_PATH = os.path.join(
    OUTPUT_DIR, "five_datasets", "all_dataset_predictions_v2.csv"
)
DATASET_METRIC_PATH = os.path.join(
    OUTPUT_DIR, "five_datasets", "all_dataset_metrics_v2.csv"
)
COMPARE_JSON = os.path.join(OUTPUT_DIR, "phase5_compare_metrics.json")
SWEEP_JSON = os.path.join(OUTPUT_DIR, "phase5_sweep_results.json")
DEFAULT_JSON = os.path.join(OUTPUT_DIR, "phase5_metrics.json")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions() -> pd.DataFrame:
    if not os.path.exists(PRED_PATH):
        raise FileNotFoundError(
            f"Missing {PRED_PATH}; run scripts.build_results_csv first."
        )
    return pd.read_csv(PRED_PATH)


def load_actions() -> pd.DataFrame:
    if not os.path.exists(ACTION_PATH):
        raise FileNotFoundError(
            f"Missing {ACTION_PATH}; run scripts.build_results_csv first."
        )
    return pd.read_csv(ACTION_PATH)


def round_numeric_df(
    df: pd.DataFrame, decimals: int = 2, exclude_cols: List[str] | None = None
) -> pd.DataFrame:
    """
    Round all numeric columns to fixed decimals, except excluded columns.
    """
    exclude_cols = set(exclude_cols or [])
    out = df.copy()
    for col in out.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(decimals)
    return out


def clean_latex_text(x):
    if isinstance(x, str):
        return x.replace("_", "-")
    return x


def sanitize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace underscores with hyphens in all text cells so LaTeX/Overleaf
    will not interpret them as subscripts.
    """
    out = df.copy()
    text_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        out[col] = out[col].apply(clean_latex_text)
    return out


def save_csv(
    df: pd.DataFrame,
    path: str,
    decimals: int = 2,
    exclude_cols: List[str] | None = None,
):
    """
    Save clean UTF-8 CSV without BOM, round numeric columns first,
    and sanitize text cells for LaTeX rendering.
    """
    out = round_numeric_df(df, decimals=decimals, exclude_cols=exclude_cols)
    out = sanitize_text_columns(out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    out.to_csv(path, index=False, encoding="utf-8")


def reset_output_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def table_i(default_metrics: Dict) -> pd.DataFrame:
    ds = default_metrics["decision_summary"]
    rows = [
        {"Metric": "Sample Count", "Value": ds.get("count")},
        {"Metric": "Accuracy (overall)", "Value": default_metrics.get("accuracy")},
        {"Metric": "Answered-only Accuracy", "Value": ds.get("accuracy_answered_only")},
        {"Metric": "AUROC", "Value": default_metrics.get("auroc")},
        {"Metric": "Answer Rate", "Value": ds.get("answer_rate")},
        {"Metric": "Abstain Rate", "Value": ds.get("abstain_rate")},
        {"Metric": "Avg. Steps", "Value": ds.get("avg_steps")},
        {"Metric": "Avg. Evidence Count", "Value": ds.get("avg_evidence")},
        {"Metric": "Avg. Budget Used", "Value": ds.get("avg_budget_used")},
        {
            "Metric": "Overconfident Error Rate",
            "Value": ds.get("overconfident_error_rate"),
        },
    ]
    return pd.DataFrame(rows)


def table_ii(compare_metrics: Dict) -> pd.DataFrame:
    rename = {
        "single_shot": "Single-Pass",
        "single_shot_rerank": "Single-Pass + Rerank",
        "single_shot_abstain": "Single-Pass + Abstain",
        "decision_loop": "Governed Decision Loop",
    }
    rows = []
    for key in [
        "single_shot",
        "single_shot_rerank",
        "single_shot_abstain",
        "decision_loop",
    ]:
        m = compare_metrics[key]
        ds = m["decision_summary"]
        rows.append(
            {
                "Method": rename[key],
                "N": ds.get("count"),
                "Acc.": m.get("accuracy"),
                "AUROC": m.get("auroc"),
                "Answer Rate": ds.get("answer_rate"),
                "Ans.-Only Acc.": ds.get("accuracy_answered_only"),
            }
        )
    return pd.DataFrame(rows)


def table_iii(sweep_metrics: Dict) -> pd.DataFrame:
    selection = [
        ("D_loose_1", "Aggressive"),
        ("F_balanced", "Balanced"),
        ("H_more_selective", "Conservative"),
    ]
    rows = []
    for key, label in selection:
        m = sweep_metrics[key]["metrics"]
        ds = m["decision_summary"]
        rows.append(
            {
                "Policy Setting": label,
                "Source Policy": key,
                "N": ds.get("count"),
                "Acc.": m.get("accuracy"),
                "AUROC": m.get("auroc"),
                "Answer Rate": ds.get("answer_rate"),
                "Ans.-Only Acc.": ds.get("accuracy_answered_only"),
                "Avg. Steps": ds.get("avg_steps"),
            }
        )
    return pd.DataFrame(rows)


def table_iii_all(sweep_metrics: Dict) -> pd.DataFrame:
    rows = []
    for key, pack in sweep_metrics.items():
        m = pack["metrics"]
        ds = m["decision_summary"]
        rows.append(
            {
                "Policy": key,
                "N": ds.get("count"),
                "Acc.": m.get("accuracy"),
                "AUROC": m.get("auroc"),
                "Answer Rate": ds.get("answer_rate"),
                "Ans.-Only Acc.": ds.get("accuracy_answered_only"),
                "Avg. Steps": ds.get("avg_steps"),
            }
        )
    return pd.DataFrame(rows)


def table_iv(pred_df: pd.DataFrame, action_df: pd.DataFrame) -> pd.DataFrame:
    default_pred = pred_df[pred_df["policy"] == "phase5_default"].copy()
    default_act = action_df[action_df["policy"] == "phase5_default"].copy()

    per_q = default_act.groupby("qid")["action"].agg(list)
    retrieve_counts = per_q.apply(lambda xs: sum(1 for x in xs if x == "RETRIEVE_MORE"))
    reason_counts = per_q.apply(lambda xs: sum(1 for x in xs if x == "RERANK"))

    rows = [
        {"Metric": "Avg. Retrieve Actions", "Value": retrieve_counts.mean()},
        {"Metric": "Avg. Reason Actions", "Value": reason_counts.mean()},
        {"Metric": "Avg. Termination Depth", "Value": default_pred["steps"].mean()},
        {
            "Metric": "Avg. Evidence per Answered Case",
            "Value": default_pred.loc[
                default_pred["answered"] == 1, "num_evidence"
            ].mean(),
        },
        {
            "Metric": "Avg. Evidence per Abstained Case",
            "Value": default_pred.loc[
                default_pred["answered"] == 0, "num_evidence"
            ].mean(),
        },
        {
            "Metric": "Avg. Budget per Sample",
            "Value": default_pred["budget_used"].mean(),
        },
        {"Metric": "Max Observed Step Depth", "Value": default_pred["steps"].max()},
    ]
    return pd.DataFrame(rows)


def table_v(pred_df: pd.DataFrame, action_df: pd.DataFrame) -> pd.DataFrame:
    default_pred = pred_df[pred_df["policy"] == "phase5_default"].copy()
    default_act = action_df[action_df["policy"] == "phase5_default"].copy()

    columns = [
        "Case",
        "Question",
        "Step",
        "Action",
        "Uncertainty",
        "Outcome / Note",
        "Final Outcome",
    ]

    success = default_pred[
        (default_pred["answered"] == 1)
        & (default_pred["correct"] == 1)
        & (default_pred["steps"] >= 2)
    ].head(1)

    abstain = default_pred[
        (default_pred["final_action"] == "ABSTAIN") & (default_pred["steps"] >= 2)
    ].head(1)

    chosen = []
    if not success.empty:
        chosen.append(("Case A", success.iloc[0]["qid"]))
    else:
        correct = default_pred[
            (default_pred["answered"] == 1) & (default_pred["correct"] == 1)
        ].head(1)
        if not correct.empty:
            chosen.append(("Case A", correct.iloc[0]["qid"]))

    if not abstain.empty:
        chosen.append(("Case B", abstain.iloc[0]["qid"]))
    else:
        failure = default_pred[
            (default_pred["answered"] == 1)
            & (default_pred["correct"] == 0)
            & (default_pred["steps"] >= 2)
        ].sort_values("uncertainty", ascending=False).head(1)
        if not failure.empty:
            chosen.append(("Case B", failure.iloc[0]["qid"]))

    rows: List[Dict] = []
    for case_label, qid in chosen:
        q_meta = default_pred[default_pred["qid"] == qid].iloc[0]
        q_actions = default_act[default_act["qid"] == qid].sort_values(
            ["history_index", "step"]
        )

        for _, r in q_actions.iterrows():
            act = str(r.get("action"))
            note = ""
            if act == "RETRIEVE_MORE":
                note = "Initial / additional evidence retrieved"
            elif act == "RERANK":
                note = "Hypothesis refined"
            elif act == "STOP":
                note = "Termination decision triggered"
            elif act == "ANSWER":
                note = "Answer committed"

            rows.append(
                {
                    "Case": case_label,
                    "Question": q_meta["question"],
                    "Step": int(r["step"]) if pd.notna(r["step"]) else None,
                    "Action": act,
                    "Uncertainty": r.get("total_uncertainty"),
                    "Outcome / Note": note,
                    "Final Outcome": (
                        q_meta["final_action"]
                        if act == q_actions.iloc[-1]["action"]
                        else ""
                    ),
                }
            )

    return pd.DataFrame(rows, columns=columns)


def figure_data_coverage(pred_df: pd.DataFrame) -> pd.DataFrame:
    sweep = pred_df[pred_df["experiment_group"] == "policy_sweep"].copy()
    out = (
        sweep.groupby(["policy", "policy_display"], dropna=False)
        .agg(
            answer_rate=("answered", "mean"),
            answered_only_accuracy=("answered_correct", "mean"),
            overall_accuracy=("correct", "mean"),
        )
        .reset_index()
        .sort_values("answer_rate", ascending=False)
    )
    return out


def figure_data_steps(pred_df: pd.DataFrame) -> pd.DataFrame:
    default_pred = pred_df[pred_df["policy"] == "phase5_default"].copy()
    out = (
        default_pred.groupby("steps", dropna=False)
        .agg(
            accuracy=("correct", "mean"),
            answer_rate=("answered", "mean"),
            count=("qid", "count"),
        )
        .reset_index()
        .sort_values("steps")
    )
    return out


def figure_data_uncertainty_hist(fd_pred: pd.DataFrame) -> pd.DataFrame:
    out = fd_pred[["dataset", "correct", "uncertainty"]].copy()
    out["label"] = out["correct"].map({1: "Correct", 0: "Incorrect"})
    return out


def main():
    reset_output_dir(TABLE_DIR)
    reset_output_dir(FIGURE_DIR)

    pred_df = load_predictions()
    action_df = load_actions()
    fd_pred = (
        pd.read_csv(DATASET_PRED_PATH)
        if os.path.exists(DATASET_PRED_PATH)
        else pd.DataFrame()
    )
    fd_metrics = (
        pd.read_csv(DATASET_METRIC_PATH)
        if os.path.exists(DATASET_METRIC_PATH)
        else pd.DataFrame()
    )
    compare_metrics = load_json(COMPARE_JSON)
    sweep_metrics = load_json(SWEEP_JSON)
    default_metrics = load_json(DEFAULT_JSON)

    t1 = table_i(default_metrics)
    t2 = table_ii(compare_metrics)
    t3 = table_iii(sweep_metrics)
    t3_all = table_iii_all(sweep_metrics)
    t4 = table_iv(pred_df, action_df)
    t5 = table_v(pred_df, action_df)

    save_csv(t1, os.path.join(TABLE_DIR, "table1_default_metrics.csv"))
    save_csv(t2, os.path.join(TABLE_DIR, "table2_baseline_comparison.csv"))
    save_csv(t3, os.path.join(TABLE_DIR, "table3_policy_sensitivity_recommended.csv"))
    save_csv(t3_all, os.path.join(TABLE_DIR, "table3_policy_sensitivity_all.csv"))
    save_csv(t4, os.path.join(TABLE_DIR, "table4_runtime_action_stats.csv"))

    # Table V 里 Step 保持整数，Uncertainty 保留两位小数
    save_csv(
        t5,
        os.path.join(TABLE_DIR, "table5_case_trajectories.csv"),
        exclude_cols=["Step"],
    )

    if not fd_metrics.empty:
        save_csv(
            fd_metrics, os.path.join(FIGURE_DIR, "figure2_dataset_roc_metrics.csv")
        )
    save_csv(
        figure_data_uncertainty_hist(fd_pred),
        os.path.join(FIGURE_DIR, "figure3_uncertainty_distribution.csv"),
    )
    save_csv(
        figure_data_coverage(pred_df),
        os.path.join(FIGURE_DIR, "figure4_coverage_reliability.csv"),
    )
    save_csv(t3_all, os.path.join(FIGURE_DIR, "figure5_policy_accuracy.csv"))
    save_csv(
        figure_data_steps(pred_df),
        os.path.join(FIGURE_DIR, "figure6_steps_vs_accuracy.csv"),
        exclude_cols=["steps", "count"],
    )

    print("Saved paper tables to", TABLE_DIR)
    print("Saved figure source data to", FIGURE_DIR)


if __name__ == "__main__":
    main()
