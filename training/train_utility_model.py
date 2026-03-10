import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import (
    UTILITY_DATASET_PATH,
    TFIDF_VECTORIZER_PATH,
    UTILITY_MODEL_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    MAX_FEATURES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    HIDDEN_DIM,
    DROPOUT,
)
from utils.io_utils import load_json
from models.utility_predictor import UtilityMLP


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


def build_text(question: str, passage: str) -> str:
    return f"{question} [SEP] {passage}"


def main() -> None:
    set_seed(RANDOM_SEED)

    data = load_json(UTILITY_DATASET_PATH)
    texts = [build_text(d["question"], d["passage"]) for d in data]
    labels = np.array([d["label"] for d in data], dtype=np.float32)

    x_train_texts, x_val_texts, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2)
    )

    x_train = vectorizer.fit_transform(x_train_texts).toarray()
    x_val = vectorizer.transform(x_val_texts).toarray()

    TFIDF_VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TFIDF_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    train_dataset = UtilityDataset(x_train, y_train)
    val_dataset = UtilityDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UtilityMLP(
        input_dim=x_train.shape[1],
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    ).to(device)

    # criterion = nn.BCEWithLogitsLoss()
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()

    pos_weight_value = float(num_neg / max(num_pos, 1))
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Using pos_weight = {pos_weight_value:.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = -1.0

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
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = torch.sigmoid(logits)

                preds = (probs >= 0.5).long().cpu().numpy()
                val_preds.extend(preds.tolist())
                val_probs.extend(probs.cpu().numpy().tolist())
                val_targets.extend(batch_y.numpy().astype(int).tolist())

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {np.mean(train_losses):.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            UTILITY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), UTILITY_MODEL_PATH)
            print(f"Saved best model to {UTILITY_MODEL_PATH}")

    print("\nFinal validation report (best threshold = 0.5 assumption):")
    print(classification_report(val_targets, val_preds, digits=4))
    print("Training complete.")


if __name__ == "__main__":
    main()