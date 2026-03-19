import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

sns.set(style="whitegrid")

CSV_PATH = "outputs/final_results.csv"
SAVE_DIR = "figures"

os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# -----------------------------
# 1 ROC CURVE
# -----------------------------
subset = df[df["method"] == "decision_loop"].dropna(subset=["uncertainty", "correct"])

if len(subset) > 0:
    y_true = 1 - subset["correct"]
    y_score = subset["uncertainty"]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"Decision Loop AUROC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Uncertainty Error Detection ROC")
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/roc_curve.png", dpi=300)
    plt.close()


# -----------------------------
# 2 UNCERTAINTY HISTOGRAM
# -----------------------------
plt.figure()

correct_unc = df[df["correct"] == 1]["uncertainty"]
wrong_unc = df[df["correct"] == 0]["uncertainty"]

sns.histplot(correct_unc, color="green", label="Correct", kde=True)
sns.histplot(wrong_unc, color="red", label="Wrong", kde=True)

plt.legend()
plt.title("Uncertainty Distribution")
plt.savefig(f"{SAVE_DIR}/uncertainty_hist.png", dpi=300)
plt.close()


# -----------------------------
# 3 POLICY ACCURACY CURVE
# -----------------------------
policy_df = (
    df[df["method"] == "decision_loop"]
    .groupby("policy")
    .agg(accuracy=("correct", "mean"))
    .reset_index()
)

plt.figure()
sns.lineplot(data=policy_df, x="policy", y="accuracy", marker="o")
plt.xticks(rotation=45)
plt.title("Accuracy under Different Decision Policies")
plt.savefig(f"{SAVE_DIR}/policy_accuracy.png", dpi=300)
plt.close()


# -----------------------------
# 4 COVERAGE vs RELIABILITY
# -----------------------------
cov_df = (
    df[df["method"] == "decision_loop"]
    .groupby("policy")
    .agg(
        answer_rate=("answered", "mean"),
        answered_accuracy=("correct", "mean")
    )
    .reset_index()
)

plt.figure()
sns.scatterplot(data=cov_df, x="answer_rate", y="answered_accuracy", hue="policy", s=100)

for i,row in cov_df.iterrows():
    plt.text(row["answer_rate"], row["answered_accuracy"], row["policy"])

plt.title("Coverage vs Reliability Trade-off")
plt.savefig(f"{SAVE_DIR}/coverage_reliability.png", dpi=300)
plt.close()


# -----------------------------
# 5 STEPS vs ACCURACY
# -----------------------------
step_df = (
    df[df["method"] == "decision_loop"]
    .groupby("steps")
    .agg(accuracy=("correct","mean"))
    .reset_index()
)

plt.figure()
sns.lineplot(data=step_df, x="steps", y="accuracy", marker="o")
plt.title("Reasoning Depth vs Accuracy")
plt.savefig(f"{SAVE_DIR}/steps_accuracy.png", dpi=300)
plt.close()


# -----------------------------
# 6 THRESHOLD SWEEP (如果有threshold列)
# -----------------------------
if "threshold" in df.columns:

    th_df = (
        df.groupby("threshold")
        .agg(accuracy=("correct","mean"))
        .reset_index()
    )

    plt.figure()
    sns.lineplot(data=th_df, x="threshold", y="accuracy", marker="o")
    plt.title("Accuracy vs Uncertainty Threshold")
    plt.savefig(f"{SAVE_DIR}/threshold_accuracy.png", dpi=300)
    plt.close()


print("All figures saved to:", SAVE_DIR)