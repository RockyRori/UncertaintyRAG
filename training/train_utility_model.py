import argparse
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from config import (
    UTILITY_DATASET_PATH,
    TFIDF_VECTORIZER_PATH,
    UTILITY_MODEL_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    MAX_FEATURES,
    BATCH_SIZE,
    EPOCHS,
    DEFAULT_TRAIN_QUESTION_LIMIT,
    LEARNING_RATE,
    HIDDEN_DIM,
    DROPOUT,
)
from models.utility_predictor import UtilityMLP
from utils.io_utils import load_json
from utils.feature_utils import (
    SAFE_STRUCTURED_FEATURE_NAMES,
    build_text_feature_from_sample,
    structured_features_from_sample,
)


# -----------------------------
# threshold search config
# -----------------------------
THRESHOLD_CANDIDATES = np.arange(0.05, 1.00, 0.05)


def limit_by_question_count(data: List[Dict], max_questions: int | None) -> List[Dict]:
    if max_questions is None or max_questions <= 0:
        return data

    selected = []
    seen_questions = set()
    for row in data:
        qid = row.get("question_id") or row.get("question") or len(seen_questions)
        if qid not in seen_questions:
            if len(seen_questions) >= max_questions:
                break
            seen_questions.add(qid)
        selected.append(row)
    return selected


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class UtilityDataset(Dataset):
    def __init__(self, x_array: np.ndarray, y_array: np.ndarray):
        self.x = torch.tensor(x_array, dtype=torch.float32)
        self.y = torch.tensor(y_array, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_text_feature(sample: Dict) -> str:
    return build_text_feature_from_sample(sample)


def extract_structured_features(sample: Dict) -> List[float]:
    return structured_features_from_sample(sample, SAFE_STRUCTURED_FEATURE_NAMES)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    best_acc = 0.0

    for threshold in THRESHOLD_CANDIDATES:
        preds = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        acc = accuracy_score(y_true, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_threshold = float(threshold)

    return best_threshold, best_f1, best_acc


def evaluate_predictions(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> None:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            auroc = roc_auc_score(y_true, y_score)
        except ValueError:
            auroc = float("nan")
    else:
        auroc = float("nan")

    print(f"\n[{name}]")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"AUROC    : {auroc:.4f}")


def run_baselines(
    y_val: np.ndarray,
    val_samples: List[Dict],
) -> None:
    y_true = y_val.astype(int)

    # baseline 1: all zero
    preds_zero = np.zeros_like(y_true)
    evaluate_predictions("Baseline / All Zero", y_true, preds_zero, preds_zero.astype(float))

    # baseline 2: all one
    preds_one = np.ones_like(y_true)
    evaluate_predictions("Baseline / All One", y_true, preds_one, preds_one.astype(float))

    # Diagnostic only: this uses gold-derived support metadata from the utility
    # dataset and is not a deployable baseline.
    support_scores = np.array(
        [int(sample.get("support", 0)) for sample in val_samples],
        dtype=int,
    )
    support_preds = support_scores.copy()
    evaluate_predictions("Diagnostic / Gold Support Rule", y_true, support_preds, support_scores.astype(float))

    # baseline 4: bm25 threshold
    bm25_scores = np.array(
        [float(sample.get("bm25_score", sample.get("score", 0.0))) for sample in val_samples],
        dtype=float,
    )

    if np.allclose(bm25_scores, bm25_scores[0]):
        print("\n[Baseline / BM25 Threshold]")
        print("Skipped: bm25_score is missing or constant in validation set.")
    else:
        bm25_threshold, bm25_best_f1, _ = find_best_threshold(y_true, bm25_scores)
        bm25_preds = (bm25_scores >= bm25_threshold).astype(int)

        print("\n[Baseline / BM25 Threshold]")
        print(f"Best threshold on val: {bm25_threshold:.4f}")
        print(f"Best F1 on val       : {bm25_best_f1:.4f}")
        evaluate_predictions(
            "Baseline / BM25 Threshold (Applied)",
            y_true,
            bm25_preds,
            bm25_scores,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-questions",
        type=int,
        default=DEFAULT_TRAIN_QUESTION_LIMIT,
        help="Limit utility training rows to the first N unique questions.",
    )
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    data = load_json(UTILITY_DATASET_PATH)
    if not data:
        raise ValueError(f"No data found in {UTILITY_DATASET_PATH}")
    data = limit_by_question_count(data, args.max_questions)
    if not data:
        raise ValueError("No utility training rows remain after applying --max-questions")
    print(f"Loaded {len(data)} utility rows after max_questions={args.max_questions}")

    labels = np.array([int(d["label"]) for d in data], dtype=np.float32)
    unique_labels, label_counts = np.unique(labels.astype(int), return_counts=True)
    stratify_labels = labels if len(unique_labels) > 1 and int(label_counts.min()) >= 2 else None

    train_samples, val_samples, y_train, y_val = train_test_split(
        data,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify_labels,
    )

    # -----------------------------
    # baseline evaluation first
    # -----------------------------
    print("============================================================")
    print("Baseline Evaluation")
    print("============================================================")
    run_baselines(y_val, val_samples)

    # -----------------------------
    # text features
    # -----------------------------
    x_train_texts = [build_text_feature(d) for d in train_samples]
    x_val_texts = [build_text_feature(d) for d in val_samples]

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        lowercase=True,
    )

    x_train_text = vectorizer.fit_transform(x_train_texts)
    x_val_text = vectorizer.transform(x_val_texts)

    # -----------------------------
    # structured features
    # -----------------------------
    x_train_struct = np.array([extract_structured_features(d) for d in train_samples], dtype=np.float32)
    x_val_struct = np.array([extract_structured_features(d) for d in val_samples], dtype=np.float32)

    scaler = StandardScaler()
    x_train_struct = scaler.fit_transform(x_train_struct)
    x_val_struct = scaler.transform(x_val_struct)

    x_train_struct_sparse = csr_matrix(x_train_struct)
    x_val_struct_sparse = csr_matrix(x_val_struct)

    # -----------------------------
    # combined features
    # -----------------------------
    x_train_all = hstack([x_train_text, x_train_struct_sparse]).astype(np.float32)
    x_val_all = hstack([x_val_text, x_val_struct_sparse]).astype(np.float32)

    # save preprocessing bundle
    TFIDF_VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TFIDF_VECTORIZER_PATH, "wb") as f:
        pickle.dump(
            {
                "vectorizer": vectorizer,
                "scaler": scaler,
                "structured_feature_names": SAFE_STRUCTURED_FEATURE_NAMES,
            },
            f,
        )

    x_train_dense = x_train_all.toarray()
    x_val_dense = x_val_all.toarray()

    train_dataset = UtilityDataset(x_train_dense, y_train)
    val_dataset = UtilityDataset(x_val_dense, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n============================================================")
    print("Model Training")
    print("============================================================")
    print(f"Using device: {device}")

    model = UtilityMLP(
        input_dim=x_train_dense.shape[1],
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)

    num_neg = float((y_train == 0).sum())
    num_pos = float((y_train == 1).sum())
    pos_weight_value = num_neg / max(num_pos, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Using pos_weight = {pos_weight_value:.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = -1.0
    best_threshold = 0.5
    best_epoch = 0

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_probs = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = torch.sigmoid(logits)

                val_probs.extend(probs.cpu().numpy().tolist())
                val_targets.extend(batch_y.numpy().astype(int).tolist())

        val_probs = np.array(val_probs, dtype=float)
        val_targets = np.array(val_targets, dtype=int)

        epoch_threshold, epoch_best_f1, epoch_best_acc = find_best_threshold(val_targets, val_probs)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {np.mean(train_losses):.4f} | "
            f"Val Best Acc: {epoch_best_acc:.4f} | "
            f"Val Best F1: {epoch_best_f1:.4f} | "
            f"Best Thresh: {epoch_threshold:.2f}"
        )

        if epoch_best_f1 > best_f1:
            best_f1 = epoch_best_f1
            best_threshold = epoch_threshold
            best_epoch = epoch + 1

            UTILITY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), UTILITY_MODEL_PATH)
            print(f"Saved best model to {UTILITY_MODEL_PATH}")

    print("\n============================================================")
    print("Best Model Evaluation")
    print("============================================================")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Best threshold: {best_threshold:.2f}")

    best_model = UtilityMLP(
        input_dim=x_train_dense.shape[1],
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)
    best_model.load_state_dict(torch.load(UTILITY_MODEL_PATH, map_location=device))
    best_model.eval()

    with torch.no_grad():
        x_val_tensor = torch.tensor(x_val_dense, dtype=torch.float32).to(device)
        final_logits = best_model(x_val_tensor)
        final_probs = torch.sigmoid(final_logits).cpu().numpy()

    final_preds = (final_probs >= best_threshold).astype(int)

    final_acc = accuracy_score(y_val, final_preds)
    final_f1 = f1_score(y_val, final_preds, zero_division=0)

    try:
        final_auroc = roc_auc_score(y_val, final_probs)
    except ValueError:
        final_auroc = float("nan")

    print("\nFinal validation metrics (best threshold search):")
    print(f"Accuracy : {final_acc:.4f}")
    print(f"F1       : {final_f1:.4f}")
    print(f"AUROC    : {final_auroc:.4f}")
    print(f"Threshold: {best_threshold:.2f}")

    print("\nFinal validation report:")
    print(classification_report(y_val, final_preds, digits=4, zero_division=0))
    print("Training complete.")


if __name__ == "__main__":
    main()
