import pandas as pd
from pathlib import Path

ROOT = Path("outputs/five_datasets")
OUT = Path("outputs/roc_source.csv")

all_rows = []

for dataset_dir in ROOT.iterdir():
    if not dataset_dir.is_dir():
        continue

    csv_path = dataset_dir / "test_predictions.csv"
    if not csv_path.exists():
        continue

    df = pd.read_csv(csv_path)

    df2 = pd.DataFrame({
        "dataset": dataset_dir.name,
        "uncertainty": df["uncertainty"],
        "label": 1 - df["correct"]   # ROC需要：1=错误
    })

    all_rows.append(df2)

roc_df = pd.concat(all_rows, ignore_index=True)
roc_df.to_csv(OUT, index=False)

print("Saved ROC source:", OUT)