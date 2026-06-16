import argparse
import csv
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from config import (
    BATCH_SIZE,
    CORPUS_PATH,
    DATA_DIR,
    DEFAULT_MULTI_DATASET_TEST_SIZE,
    DEFAULT_MULTI_DATASET_TRAIN_SIZE,
    DROPOUT,
    EPOCHS,
    GENERATOR_MODEL_NAME,
    HIDDEN_DIM,
    LEARNING_RATE,
    MAX_FEATURES,
    MAX_INPUT_LENGTH,
    MAX_NEW_TOKENS,
    MAX_RETRIEVAL_BUDGET,
    MAX_DECISION_STEPS,
    RANDOM_SEED,
    RETRIEVE_MORE_K,
    SAVED_MODELS_DIR,
    OUTPUTS_DIR,
    TAU_ANSWER,
    TAU_CONFLICT,
    TAU_DELTA,
    TAU_GAIN,
    TAU_RETRIEVE,
    TAU_STOP,
    UNCERTAINTY_ALPHA,
    UNCERTAINTY_BETA,
    UNCERTAINTY_GAMMA,
    INITIAL_TOP_K,
)
from controller.policy import RuleBasedPolicy
from decision.loop import DecisionAwareRAG
from evaluation.metrics import compute_accuracy, compute_auroc, compute_avg_uncertainty, selective_accuracy
from generator.simple_answerer import SimpleAnswerer
from inference.predict_utility import UtilityPredictor
from models.utility_predictor import UtilityMLP
from retriever.bm25_retriever import BM25Retriever
from training.train_utility_model import (
    UtilityDataset,
    build_text_feature,
    extract_structured_features,
    find_best_threshold,
    set_seed,
)
from uncertainty.signals import DecisionAwareUncertainty
from utils.io_utils import load_json, save_json
from utils.feature_utils import SAFE_STRUCTURED_FEATURE_NAMES

DATASET_NAMES = ["nq", "triviaqa", "webq", "squad", "popqa"]


class DatasetFiles:
    def __init__(
        self,
        base_dir: Path,
        dataset_name: str,
        train_size: int = DEFAULT_MULTI_DATASET_TRAIN_SIZE,
        test_size: int = DEFAULT_MULTI_DATASET_TEST_SIZE,
    ):
        self.dataset_name = dataset_name
        self.qa_path = base_dir / "processed" / f"{dataset_name}_qa.json"
        self.corpus_path = base_dir / "processed" / f"{dataset_name}_corpus.json"
        self.mini_dir = base_dir / "mini" / dataset_name
        self.mini_train_path = self.mini_dir / f"train_{train_size}.json"
        self.mini_test_path = self.mini_dir / f"test_{test_size}.json"
        self.utility_dataset_path = self.mini_dir / "utility_train.json"
        self.model_dir = SAVED_MODELS_DIR / dataset_name
        self.model_path = self.model_dir / "utility_mlp.pt"
        self.vectorizer_path = self.model_dir / "tfidf_vectorizer.pkl"
        self.output_dir = OUTPUTS_DIR / "five_datasets" / dataset_name
        self.predictions_csv = self.output_dir / "test_predictions.csv"
        self.metrics_json = self.output_dir / "metrics.json"


class CSVDictWriter:
    def __init__(self, path: Path, fieldnames: List[str]):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames

    def write(self, rows: List[Dict]):
        with self.path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def load_dataset_records(qa_path: Path) -> List[Dict]:
    if not qa_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {qa_path}")
    data = load_json(qa_path)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Dataset file is empty or invalid: {qa_path}")
    return data


def get_split_records(records: List[Dict], split_name: str) -> List[Dict]:
    return [x for x in records if str(x.get("split", "")).lower() == split_name]


def build_fixed_mini_splits(
    dataset_name: str,
    files: DatasetFiles,
    train_n: int = DEFAULT_MULTI_DATASET_TRAIN_SIZE,
    test_n: int = DEFAULT_MULTI_DATASET_TEST_SIZE,
) -> Tuple[List[Dict], List[Dict]]:
    records = load_dataset_records(files.qa_path)

    train_records = get_split_records(records, "train")
    test_records = get_split_records(records, "test")
    dev_records = get_split_records(records, "dev")

    rng = random.Random(RANDOM_SEED)

    # 优先使用真实 split；没有 test 时用 dev 顶上；还没有就从 all 里做伪切分
    if len(train_records) >= train_n:
        train_selected = rng.sample(train_records, train_n)
    else:
        fallback_source = train_records if train_records else records
        if len(fallback_source) < train_n:
            raise ValueError(f"{dataset_name}: not enough samples to build {train_n}-train split")
        train_selected = rng.sample(fallback_source, train_n)

    if len(test_records) >= test_n:
        test_selected = rng.sample(test_records, test_n)
    elif len(dev_records) >= test_n:
        test_selected = rng.sample(dev_records, test_n)
    else:
        remaining = [x for x in records if x.get("id") not in {r.get("id") for r in train_selected}]
        if len(remaining) < test_n:
            remaining = [x for x in records if x.get("id") not in set()]
        if len(remaining) < test_n:
            raise ValueError(f"{dataset_name}: not enough samples to build {test_n}-test split")
        test_selected = rng.sample(remaining, test_n)

    files.mini_dir.mkdir(parents=True, exist_ok=True)
    save_json(train_selected, files.mini_train_path)
    save_json(test_selected, files.mini_test_path)
    return train_selected, test_selected


class MiniUtilityBuilder:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    @staticmethod
    def extract_passage_text(item: Dict) -> str:
        for key in ["text", "passage", "content", "body", "context"]:
            if key in item:
                return str(item[key])
        return str(item)

    @staticmethod
    def normalize(text: str) -> str:
        return str(text or "").strip().lower()

    def build(self, samples: List[Dict], top_k: int = 5) -> List[Dict]:
        rows = []
        for sample_idx, sample in enumerate(samples):
            question = sample["question"]
            gold_answers = sample.get("gold_answers", [])
            qid = sample.get("id", f"q_{sample_idx}")
            retrieved = self.retriever.retrieve(question, top_k=top_k)

            for passage_idx, item in enumerate(retrieved):
                passage = self.extract_passage_text(item)
                pred_answer = self.generator.answer_with_single_passage(question, passage)
                pred_norm = self.normalize(pred_answer)
                passage_norm = self.normalize(passage)
                answer_correct = int(any(self.normalize(g) == pred_norm or self.normalize(g) in pred_norm for g in gold_answers if self.normalize(g)))
                support = int(any(self.normalize(g) in passage_norm for g in gold_answers if self.normalize(g)))
                rows.append({
                    "question_id": qid,
                    "question": question,
                    "gold_answers": gold_answers,
                    "passage_id": item.get("id", f"{qid}_p_{passage_idx}"),
                    "passage_index": passage_idx,
                    "passage_rank": passage_idx + 1,
                    "bm25_score": float(item.get("score", 0.0)),
                    "passage": passage,
                    "pred_answer": pred_answer,
                    "pred_answer_in_passage": int(pred_norm in passage_norm) if pred_norm else 0,
                    "answer_correct": answer_correct,
                    "support": support,
                    "utility_score": float(answer_correct),
                    "label": answer_correct,
                })
        return rows


class UtilityTrainer:
    def __init__(self, model_path: Path, vectorizer_path: Path):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path

    def fit(self, data: List[Dict]) -> Dict:
        labels = np.array([int(d["label"]) for d in data], dtype=np.float32)
        unique_labels, label_counts = np.unique(labels.astype(int), return_counts=True)
        stratify_labels = labels if len(unique_labels) > 1 and int(label_counts.min()) >= 2 else None

        train_samples, val_samples, y_train, y_val = train_test_split(
            data,
            labels,
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=stratify_labels,
        )

        x_train_texts = [build_text_feature(d) for d in train_samples]
        x_val_texts = [build_text_feature(d) for d in val_samples]

        vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), lowercase=True)
        x_train_text = vectorizer.fit_transform(x_train_texts)
        x_val_text = vectorizer.transform(x_val_texts)

        x_train_struct = np.array([extract_structured_features(d) for d in train_samples], dtype=np.float32)
        x_val_struct = np.array([extract_structured_features(d) for d in val_samples], dtype=np.float32)

        scaler = StandardScaler()
        x_train_struct = scaler.fit_transform(x_train_struct)
        x_val_struct = scaler.transform(x_val_struct)

        x_train_all = hstack([x_train_text, csr_matrix(x_train_struct)]).astype(np.float32)
        x_val_all = hstack([x_val_text, csr_matrix(x_val_struct)]).astype(np.float32)

        self.vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        with self.vectorizer_path.open("wb") as f:
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

        train_loader = DataLoader(UtilityDataset(x_train_dense, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(UtilityDataset(x_val_dense, y_val), batch_size=BATCH_SIZE, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UtilityMLP(input_dim=x_train_dense.shape[1], hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)

        num_neg = float((y_train == 0).sum())
        num_pos = float((y_train == 1).sum())
        pos_weight_value = num_neg / max(num_pos, 1.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32).to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_f1 = -1.0
        best_threshold = 0.5
        best_epoch = 0

        for epoch in range(EPOCHS):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            val_probs, val_targets = [], []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    probs = torch.sigmoid(model(batch_x))
                    val_probs.extend(probs.cpu().numpy().tolist())
                    val_targets.extend(batch_y.numpy().astype(int).tolist())

            val_probs = np.array(val_probs, dtype=float)
            val_targets = np.array(val_targets, dtype=int)
            threshold, best_epoch_f1, _ = find_best_threshold(val_targets, val_probs)
            if best_epoch_f1 > best_f1:
                best_f1 = best_epoch_f1
                best_threshold = threshold
                best_epoch = epoch + 1
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), self.model_path)

        best_model = UtilityMLP(input_dim=x_train_dense.shape[1], hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
        best_model.load_state_dict(torch.load(self.model_path, map_location=device))
        best_model.eval()
        with torch.no_grad():
            final_probs = torch.sigmoid(best_model(torch.tensor(x_val_dense, dtype=torch.float32).to(device))).cpu().numpy()
        final_preds = (final_probs >= best_threshold).astype(int)
        return {
            "val_accuracy": float(accuracy_score(y_val, final_preds)),
            "val_precision": float(precision_score(y_val, final_preds, zero_division=0)),
            "val_recall": float(recall_score(y_val, final_preds, zero_division=0)),
            "val_f1": float(f1_score(y_val, final_preds, zero_division=0)),
            "val_auroc": float(roc_auc_score(y_val, final_probs)) if len(np.unique(y_val)) > 1 else None,
            "best_threshold": float(best_threshold),
            "best_epoch": int(best_epoch),
        }


def run_dataset_experiment(
    dataset_name: str,
    top_k_utility: int = 5,
    train_size: int = DEFAULT_MULTI_DATASET_TRAIN_SIZE,
    test_size: int = DEFAULT_MULTI_DATASET_TEST_SIZE,
) -> Dict:
    files = DatasetFiles(DATA_DIR, dataset_name, train_size=train_size, test_size=test_size)
    train_samples, test_samples = build_fixed_mini_splits(
        dataset_name,
        files,
        train_n=train_size,
        test_n=test_size,
    )

    corpus = load_json(files.corpus_path)
    retriever = BM25Retriever(corpus)
    utility_builder = MiniUtilityBuilder(
        retriever=retriever,
        generator=SimpleAnswerer(
            model_name=GENERATOR_MODEL_NAME,
            max_input_length=MAX_INPUT_LENGTH,
            max_new_tokens=MAX_NEW_TOKENS,
        ).generator,
    )
    utility_rows = utility_builder.build(train_samples, top_k=top_k_utility)
    save_json(utility_rows, files.utility_dataset_path)

    trainer = UtilityTrainer(files.model_path, files.vectorizer_path)
    train_metrics = trainer.fit(utility_rows)

    utility_predictor = UtilityPredictor(model_path=files.model_path, vectorizer_path=files.vectorizer_path)
    answerer = SimpleAnswerer(
        model_name=GENERATOR_MODEL_NAME,
        max_input_length=MAX_INPUT_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    uncertainty_scorer = DecisionAwareUncertainty(
        alpha=UNCERTAINTY_ALPHA,
        beta=UNCERTAINTY_BETA,
        gamma=UNCERTAINTY_GAMMA,
    )
    policy = RuleBasedPolicy(
        tau_answer=TAU_ANSWER,
        tau_retrieve=TAU_RETRIEVE,
        tau_conflict=TAU_CONFLICT,
        tau_stop=TAU_STOP,
        tau_delta=TAU_DELTA,
        tau_gain=TAU_GAIN,
    )
    runner = DecisionAwareRAG(
        retriever=retriever,
        utility_predictor=utility_predictor,
        answerer=answerer,
        uncertainty_scorer=uncertainty_scorer,
        policy=policy,
        initial_top_k=INITIAL_TOP_K,
        retrieve_more_k=RETRIEVE_MORE_K,
        max_steps=MAX_DECISION_STEPS,
        max_budget=MAX_RETRIEVAL_BUDGET,
    )

    test_records = []
    csv_rows = []
    for idx, sample in enumerate(test_samples):
        state = runner.run_one(question=sample["question"], gold_answers=sample.get("gold_answers", []))
        test_records.append({
            "question": sample["question"],
            "gold_answers": sample.get("gold_answers", []),
            "final_answer": state.final_answer,
            "final_action": state.final_action,
            "correct": state.correct,
            "uncertainty": state.total_uncertainty,
            "retrieval_uncertainty": state.retrieval_uncertainty,
            "conflict_uncertainty": state.conflict_uncertainty,
            "stability_uncertainty": state.stability_uncertainty,
            "steps": state.step,
            "num_evidence": len(state.evidence),
            "budget_used": MAX_RETRIEVAL_BUDGET - state.remaining_budget,
            "stop_reason": state.stop_reason,
            "history": state.history,
        })
        csv_rows.append({
            "dataset": dataset_name,
            "sample_id": sample.get("id", f"test_{idx}"),
            "question": sample["question"],
            "gold_answers": " || ".join(sample.get("gold_answers", [])),
            "prediction": state.final_answer,
            "final_action": state.final_action,
            "correct": int(state.correct),
            "uncertainty": round(float(state.total_uncertainty), 6),
            "retrieval_uncertainty": round(float(state.retrieval_uncertainty), 6),
            "conflict_uncertainty": round(float(state.conflict_uncertainty), 6),
            "stability_uncertainty": round(float(state.stability_uncertainty), 6),
            "steps": int(state.step),
            "num_evidence": int(len(state.evidence)),
            "budget_used": int(MAX_RETRIEVAL_BUDGET - state.remaining_budget),
            "stop_reason": state.stop_reason,
        })

    CSVDictWriter(
        files.predictions_csv,
        [
            "dataset", "sample_id", "question", "gold_answers", "prediction", "final_action", "correct",
            "uncertainty", "retrieval_uncertainty", "conflict_uncertainty", "stability_uncertainty",
            "steps", "num_evidence", "budget_used", "stop_reason",
        ],
    ).write(csv_rows)

    test_metrics = {
        "accuracy": compute_accuracy(test_records),
        "auroc": compute_auroc(test_records),
        "avg_uncertainty": compute_avg_uncertainty(test_records),
        "selective_accuracy_80": selective_accuracy(test_records, keep_ratio=0.8),
    }

    result = {
        "dataset": dataset_name,
        "train_size": len(train_samples),
        "test_size": len(test_samples),
        **train_metrics,
        **test_metrics,
        "predictions_csv": str(files.predictions_csv),
        "metrics_json": str(files.metrics_json),
    }
    save_json(result, files.metrics_json)
    return result


def write_summary_csv(summary_rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "train_size", "test_size", "val_accuracy", "val_precision", "val_recall", "val_f1", "val_auroc",
        "best_threshold", "best_epoch", "accuracy", "auroc", "avg_uncertainty_overall",
        "avg_uncertainty_correct", "avg_uncertainty_incorrect", "selective_accuracy_80",
        "kept_count_80", "predictions_csv", "metrics_json",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DATASET_NAMES)
    parser.add_argument("--top-k-utility", type=int, default=5)
    parser.add_argument("--train-size", type=int, default=DEFAULT_MULTI_DATASET_TRAIN_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_MULTI_DATASET_TEST_SIZE)
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    summary_rows = []
    for dataset_name in args.datasets:
        result = run_dataset_experiment(
            dataset_name=dataset_name,
            top_k_utility=args.top_k_utility,
            train_size=args.train_size,
            test_size=args.test_size,
        )
        summary_rows.append({
            "dataset": result["dataset"],
            "train_size": result["train_size"],
            "test_size": result["test_size"],
            "val_accuracy": result["val_accuracy"],
            "val_precision": result["val_precision"],
            "val_recall": result["val_recall"],
            "val_f1": result["val_f1"],
            "val_auroc": result["val_auroc"],
            "best_threshold": result["best_threshold"],
            "best_epoch": result["best_epoch"],
            "accuracy": result["accuracy"],
            "auroc": result["auroc"],
            "avg_uncertainty_overall": result["avg_uncertainty"]["overall"],
            "avg_uncertainty_correct": result["avg_uncertainty"]["correct_only"],
            "avg_uncertainty_incorrect": result["avg_uncertainty"]["incorrect_only"],
            "selective_accuracy_80": result["selective_accuracy_80"]["accuracy"],
            "kept_count_80": result["selective_accuracy_80"]["kept_count"],
            "predictions_csv": result["predictions_csv"],
            "metrics_json": result["metrics_json"],
        })

    summary_path = OUTPUTS_DIR / "five_datasets" / "summary_metrics.csv"
    write_summary_csv(summary_rows, summary_path)
    print(f"Saved summary csv to {summary_path}")


if __name__ == "__main__":
    main()
