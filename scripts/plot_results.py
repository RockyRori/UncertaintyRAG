import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve

from config import OUTPUTS_DIR

OUTPUT_DIR = str(OUTPUTS_DIR)
SAVE_DIR = os.path.join(OUTPUT_DIR, "figures_v2")
PRED_PATH = os.path.join(OUTPUT_DIR, "final_results_v2.csv")
DATASET_PRED_PATH = os.path.join(OUTPUT_DIR, "five_datasets", "all_dataset_predictions_v2.csv")
DATASET_METRIC_PATH = os.path.join(OUTPUT_DIR, "five_datasets", "all_dataset_metrics_v2.csv")


def _safe_read(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def reset_output_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def plot_roc_by_dataset(fd_pred: pd.DataFrame, fd_metrics: pd.DataFrame):
    plt.figure(figsize=(7.5, 5.5))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Baseline")

    overall = fd_pred.dropna(subset=["uncertainty", "correct"]).copy()
    overall["error_label"] = 1 - overall["correct"]
    fpr, tpr, _ = roc_curve(overall["error_label"], overall["uncertainty"])
    plt.plot(fpr, tpr, label=f"Overall (AUROC = {auc(fpr, tpr):.3f})")

    order = ["nq", "popqa", "squad", "triviaqa", "webq"]
    label_map = {"nq": "NQ", "popqa": "PopQA", "squad": "SQuAD", "triviaqa": "TriviaQA", "webq": "WebQ"}
    for ds in order:
        subset = fd_pred[fd_pred["dataset"] == ds].dropna(subset=["uncertainty", "correct"]).copy()
        if subset.empty:
            continue
        subset["error_label"] = 1 - subset["correct"]
        fpr, tpr, _ = roc_curve(subset["error_label"], subset["uncertainty"])
        au = fd_metrics.loc[fd_metrics["dataset"] == ds, "auroc"]
        au_text = float(au.iloc[0]) if len(au) else auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label_map.get(ds, ds)} (AUROC = {au_text:.3f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Uncertainty-Based Error Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "figure2_roc_curves.png"), dpi=300)
    plt.close()


def plot_uncertainty_distribution(fd_pred: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    correct = fd_pred.loc[fd_pred["correct"] == 1, "uncertainty"].dropna()
    incorrect = fd_pred.loc[fd_pred["correct"] == 0, "uncertainty"].dropna()
    plt.hist(correct, bins=20, alpha=0.6, density=True, label="Correct")
    plt.hist(incorrect, bins=20, alpha=0.6, density=True, label="Incorrect")
    plt.xlabel("Uncertainty")
    plt.ylabel("Density")
    plt.title("Distribution of Uncertainty Scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "figure3_uncertainty_distribution.png"), dpi=300)
    plt.close()


def plot_coverage_reliability(pred_df: pd.DataFrame):
    sweep = pred_df[pred_df["experiment_group"] == "policy_sweep"].copy()
    summary = (
        sweep.groupby(["policy", "policy_display"], dropna=False)
        .agg(answer_rate=("answered", "mean"), answered_only_accuracy=("answered_correct", "mean"))
        .reset_index()
        .sort_values("answer_rate", ascending=False)
    )

    plt.figure(figsize=(7, 5))
    plt.scatter(summary["answer_rate"], summary["answered_only_accuracy"])
    for _, row in summary.iterrows():
        plt.text(row["answer_rate"], row["answered_only_accuracy"], str(row["policy_display"]))
    plt.xlabel("Coverage / Answer Rate")
    plt.ylabel("Answered-only Accuracy")
    plt.title("Coverage–Reliability Trade-off")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "figure4_coverage_reliability.png"), dpi=300)
    plt.close()


def plot_policy_accuracy(pred_df: pd.DataFrame):
    sweep = pred_df[pred_df["experiment_group"] == "policy_sweep"].copy()
    summary = (
        sweep.groupby(["policy", "policy_display"], dropna=False)
        .agg(accuracy=("correct", "mean"), answered_only_accuracy=("answered_correct", "mean"), answer_rate=("answered", "mean"))
        .reset_index()
        .sort_values("answer_rate", ascending=False)
    )

    plt.figure(figsize=(8, 5))
    x = range(len(summary))
    plt.plot(x, summary["accuracy"], marker="o", label="Overall Accuracy")
    plt.plot(x, summary["answered_only_accuracy"], marker="o", label="Answered-only Accuracy")
    plt.xticks(list(x), summary["policy_display"], rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Accuracy under Different Governance Policies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "figure5_policy_accuracy.png"), dpi=300)
    plt.close()


def plot_steps_vs_accuracy(pred_df: pd.DataFrame):
    default_pred = pred_df[pred_df["policy"] == "phase5_default"].copy()
    summary = (
        default_pred.groupby("steps", dropna=False)
        .agg(accuracy=("correct", "mean"), answer_rate=("answered", "mean"), count=("qid", "count"))
        .reset_index()
        .sort_values("steps")
    )

    plt.figure(figsize=(7, 5))
    plt.plot(summary["steps"], summary["accuracy"], marker="o", label="Accuracy")
    plt.plot(summary["steps"], summary["answer_rate"], marker="o", label="Answer Rate")
    for _, row in summary.iterrows():
        plt.text(row["steps"], row["accuracy"], f"n={int(row['count'])}")
    plt.xlabel("Decision Steps")
    plt.ylabel("Score")
    plt.title("Decision Depth vs Accuracy-Related Outcomes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "figure6_steps_vs_accuracy.png"), dpi=300)
    plt.close()


def main():
    reset_output_dir(SAVE_DIR)

    pred_df = _safe_read(PRED_PATH)
    fd_pred = _safe_read(DATASET_PRED_PATH)
    fd_metrics = _safe_read(DATASET_METRIC_PATH)

    plot_roc_by_dataset(fd_pred, fd_metrics)
    plot_uncertainty_distribution(fd_pred)
    plot_coverage_reliability(pred_df)
    plot_policy_accuracy(pred_df)
    plot_steps_vs_accuracy(pred_df)
    print(f"All paper figures saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
