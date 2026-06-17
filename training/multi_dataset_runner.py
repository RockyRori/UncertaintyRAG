import argparse
import csv
import pickle
import random
import shutil
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
    HIDDEN_DIM,
    LEARNING_RATE,
    MAX_FEATURES,
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
    ANSWER_MIN_UTILITY,
    ANSWER_MIN_BEST_UTILITY,
    ANSWER_MAX_CONFLICT,
    ANSWER_MAX_TOTAL_UNCERTAINTY,
    ANSWER_MIN_STABILITY,
    ANSWER_HIGH_UTILITY,
    ANSWER_HIGH_UTILITY_MIN_ANSWER_UTILITY,
    ANSWER_HIGH_UTILITY_MAX_TOTAL_UNCERTAINTY,
    UNCERTAINTY_ALPHA,
    UNCERTAINTY_BETA,
    UNCERTAINTY_GAMMA,
    INITIAL_TOP_K,
)
from controller.policy import RuleBasedPolicy
from decision.loop import DecisionAwareRAG
from evaluation.metrics import compute_accuracy, compute_auroc, compute_avg_uncertainty, selective_accuracy
from evaluation.metrics import compute_answer_metrics
from generator.deepseek_answerer import DeepSeekAnswerer
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
from utils.text_utils import qa_metrics, relaxed_match_score
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


def extract_corpus_text(item: Dict) -> str:
    for key in ["text", "passage", "content", "body", "context"]:
        if key in item:
            return str(item.get(key) or "")
    return str(item)


def corpus_quality_report(corpus: List[Dict]) -> Dict:
    placeholder_markers = [
        "no gold answer",
        "external retrieval corpus",
        "no object or possible_answers",
        "placeholder",
    ]
    placeholder_count = 0
    empty_count = 0
    for item in corpus:
        text = extract_corpus_text(item).strip().lower()
        if not text:
            empty_count += 1
        if any(marker in text for marker in placeholder_markers):
            placeholder_count += 1

    total = len(corpus)
    return {
        "corpus_count": total,
        "empty_count": empty_count,
        "placeholder_count": placeholder_count,
        "placeholder_ratio": placeholder_count / total if total else 1.0,
    }


def validate_corpus_quality(dataset_name: str, corpus: List[Dict]) -> Dict:
    report = corpus_quality_report(corpus)
    if report["placeholder_ratio"] >= 0.5:
        raise ValueError(
            f"{dataset_name}: placeholder corpus ratio is too high "
            f"({report['placeholder_ratio']:.2f}); provide an external retrieval corpus "
            "or rerun with --allow-placeholder-corpus for stress testing only"
        )
    return report


def get_split_records(records: List[Dict], split_name: str) -> List[Dict]:
    return [x for x in records if str(x.get("split", "")).lower() == split_name]


def record_key(record: Dict) -> str:
    if record.get("id") is not None:
        return str(record.get("id"))
    return f"{record.get('split', '')}::{record.get('question', '')}"


def sample_without_replacement(
    rng: random.Random,
    records: List[Dict],
    n: int,
    dataset_name: str,
    split_label: str,
) -> List[Dict]:
    if len(records) < n:
        raise ValueError(
            f"{dataset_name}: not enough non-overlapping {split_label} samples "
            f"({len(records)} available, {n} requested)"
        )
    return rng.sample(records, n)


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
    if train_records:
        train_selected = sample_without_replacement(
            rng, train_records, train_n, dataset_name, "train"
        )
    else:
        train_selected = sample_without_replacement(
            rng, records, train_n, dataset_name, "train fallback"
        )

    train_ids = {record_key(r) for r in train_selected}
    test_candidates = [r for r in test_records if record_key(r) not in train_ids]
    dev_candidates = [r for r in dev_records if record_key(r) not in train_ids]
    remaining_candidates = [r for r in records if record_key(r) not in train_ids]

    if len(test_candidates) >= test_n:
        test_selected = rng.sample(test_candidates, test_n)
    elif len(dev_candidates) >= test_n:
        test_selected = rng.sample(dev_candidates, test_n)
    else:
        test_selected = sample_without_replacement(
            rng, remaining_candidates, test_n, dataset_name, "test fallback"
        )

    overlap = {record_key(r) for r in train_selected} & {
        record_key(r) for r in test_selected
    }
    if overlap:
        raise ValueError(
            f"{dataset_name}: train/test split overlap detected "
            f"({len(overlap)} duplicated ids)"
        )

    files.mini_dir.mkdir(parents=True, exist_ok=True)
    save_json(train_selected, files.mini_train_path)
    save_json(test_selected, files.mini_test_path)
    return train_selected, test_selected


class MiniUtilityBuilder:
    def __init__(self, retriever, answerer):
        self.retriever = retriever
        self.answerer = answerer

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
                pred_answer = self.answerer.answer_with_single_passage(question, passage)
                pred_norm = self.normalize(pred_answer)
                passage_norm = self.normalize(passage)
                exact_answer_correct = int(
                    any(
                        self.normalize(g) == pred_norm
                        for g in gold_answers
                        if self.normalize(g)
                    )
                )
                answer_correct = int(relaxed_match_score(pred_answer, gold_answers))
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
                    "exact_answer_correct": exact_answer_correct,
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
    allow_placeholder_corpus: bool = False,
    external_corpus_dir: Path | None = None,
) -> Dict:
    files = DatasetFiles(DATA_DIR, dataset_name, train_size=train_size, test_size=test_size)
    train_samples, test_samples = build_fixed_mini_splits(
        dataset_name,
        files,
        train_n=train_size,
        test_n=test_size,
    )

    corpus_path = files.corpus_path
    if external_corpus_dir is not None:
        candidate = external_corpus_dir / f"{dataset_name}_corpus.json"
        if candidate.exists():
            corpus_path = candidate

    corpus = load_json(corpus_path)
    corpus_report = corpus_quality_report(corpus)
    if not allow_placeholder_corpus:
        corpus_report = validate_corpus_quality(dataset_name, corpus)
    retriever = BM25Retriever(corpus)
    answerer = DeepSeekAnswerer()
    utility_builder = MiniUtilityBuilder(
        retriever=retriever,
        answerer=answerer,
    )
    utility_rows = utility_builder.build(train_samples, top_k=top_k_utility)
    save_json(utility_rows, files.utility_dataset_path)

    trainer = UtilityTrainer(files.model_path, files.vectorizer_path)
    train_metrics = trainer.fit(utility_rows)

    utility_predictor = UtilityPredictor(model_path=files.model_path, vectorizer_path=files.vectorizer_path)
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
        answer_min_utility=ANSWER_MIN_UTILITY,
        answer_min_best_utility=ANSWER_MIN_BEST_UTILITY,
        answer_max_conflict=ANSWER_MAX_CONFLICT,
        answer_max_total_uncertainty=ANSWER_MAX_TOTAL_UNCERTAINTY,
        tau_stability=ANSWER_MIN_STABILITY,
        answer_high_utility=ANSWER_HIGH_UTILITY,
        answer_high_utility_min_answer_utility=ANSWER_HIGH_UTILITY_MIN_ANSWER_UTILITY,
        answer_high_utility_max_total_uncertainty=ANSWER_HIGH_UTILITY_MAX_TOTAL_UNCERTAINTY,
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
        answer_metrics = (
            qa_metrics(state.final_answer, sample.get("gold_answers", []))
            if state.final_action == "ANSWER"
            else {
                "exact_match": 0.0,
                "relaxed_match": 0.0,
                "contains_answer": 0.0,
                "token_f1": 0.0,
            }
        )
        test_records.append({
            "question": sample["question"],
            "gold_answers": sample.get("gold_answers", []),
            "final_answer": state.final_answer,
            "final_action": state.final_action,
            "correct": state.correct,
            **answer_metrics,
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
            "exact_match": round(float(answer_metrics["exact_match"]), 6),
            "relaxed_match": round(float(answer_metrics["relaxed_match"]), 6),
            "contains_answer": round(float(answer_metrics["contains_answer"]), 6),
            "token_f1": round(float(answer_metrics["token_f1"]), 6),
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
            "exact_match", "relaxed_match", "contains_answer", "token_f1",
            "uncertainty", "retrieval_uncertainty", "conflict_uncertainty", "stability_uncertainty",
            "steps", "num_evidence", "budget_used", "stop_reason",
        ],
    ).write(csv_rows)

    test_metrics = {
        "accuracy": compute_accuracy(test_records),
        "answer_metrics": compute_answer_metrics(test_records),
        "auroc": compute_auroc(test_records),
        "avg_uncertainty": compute_avg_uncertainty(test_records),
        "selective_accuracy_80": selective_accuracy(test_records, keep_ratio=0.8),
    }

    result = {
        "dataset": dataset_name,
        "train_size": len(train_samples),
        "test_size": len(test_samples),
        "corpus_quality": corpus_report,
        "corpus_path": str(corpus_path),
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
        "best_threshold", "best_epoch", "accuracy", "exact_match", "relaxed_match", "contains_answer", "token_f1",
        "answered_exact_match", "answered_relaxed_match", "answered_contains_answer", "answered_token_f1",
        "auroc", "avg_uncertainty_overall",
        "avg_uncertainty_correct", "avg_uncertainty_incorrect", "selective_accuracy_80",
        "kept_count_80", "corpus_placeholder_ratio", "corpus_path", "predictions_csv", "metrics_json",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def clear_dataset_outputs(dataset_name: str, train_size: int, test_size: int) -> None:
    files = DatasetFiles(DATA_DIR, dataset_name, train_size=train_size, test_size=test_size)
    if files.output_dir.exists():
        shutil.rmtree(files.output_dir)


def write_skipped_csv(skipped_rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "reason"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in skipped_rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DATASET_NAMES)
    parser.add_argument("--top-k-utility", type=int, default=5)
    parser.add_argument("--train-size", type=int, default=DEFAULT_MULTI_DATASET_TRAIN_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_MULTI_DATASET_TEST_SIZE)
    parser.add_argument(
        "--strict-splits",
        action="store_true",
        help="Fail instead of skipping datasets that cannot form non-overlapping splits.",
    )
    parser.add_argument(
        "--allow-placeholder-corpus",
        action="store_true",
        help="Allow placeholder corpora for stress testing. Do not use for main results.",
    )
    parser.add_argument(
        "--external-corpus-dir",
        type=str,
        default=None,
        help="Optional directory containing <dataset>_corpus.json files.",
    )
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    summary_rows = []
    skipped_rows = []
    for dataset_name in args.datasets:
        try:
            result = run_dataset_experiment(
                dataset_name=dataset_name,
                top_k_utility=args.top_k_utility,
                train_size=args.train_size,
                test_size=args.test_size,
                allow_placeholder_corpus=args.allow_placeholder_corpus,
                external_corpus_dir=Path(args.external_corpus_dir)
                if args.external_corpus_dir
                else None,
            )
        except ValueError as exc:
            if args.strict_splits:
                raise
            print(f"[SKIP] {dataset_name}: {exc}")
            skipped_rows.append({"dataset": dataset_name, "reason": str(exc)})
            clear_dataset_outputs(dataset_name, args.train_size, args.test_size)
            continue
        answer_metrics = result["answer_metrics"]
        corpus_quality = result.get("corpus_quality", {})
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
            "exact_match": answer_metrics["exact_match"],
            "relaxed_match": answer_metrics["relaxed_match"],
            "contains_answer": answer_metrics["contains_answer"],
            "token_f1": answer_metrics["token_f1"],
            "answered_exact_match": answer_metrics["answered_exact_match"],
            "answered_relaxed_match": answer_metrics["answered_relaxed_match"],
            "answered_contains_answer": answer_metrics["answered_contains_answer"],
            "answered_token_f1": answer_metrics["answered_token_f1"],
            "auroc": result["auroc"],
            "avg_uncertainty_overall": result["avg_uncertainty"]["overall"],
            "avg_uncertainty_correct": result["avg_uncertainty"]["correct_only"],
            "avg_uncertainty_incorrect": result["avg_uncertainty"]["incorrect_only"],
            "selective_accuracy_80": result["selective_accuracy_80"]["accuracy"],
            "kept_count_80": result["selective_accuracy_80"]["kept_count"],
            "corpus_placeholder_ratio": corpus_quality.get("placeholder_ratio"),
            "corpus_path": result.get("corpus_path"),
            "predictions_csv": result["predictions_csv"],
            "metrics_json": result["metrics_json"],
        })

    summary_path = OUTPUTS_DIR / "five_datasets" / "summary_metrics.csv"
    write_summary_csv(summary_rows, summary_path)
    skipped_path = OUTPUTS_DIR / "five_datasets" / "skipped_datasets.csv"
    write_skipped_csv(skipped_rows, skipped_path)
    print(f"Saved summary csv to {summary_path}")
    print(f"Saved skipped dataset csv to {skipped_path}")


if __name__ == "__main__":
    main()
