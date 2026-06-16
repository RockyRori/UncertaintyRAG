import argparse
import csv
from pathlib import Path
from typing import Dict, List

from config import DATASET_CORPUS_PATH, DATASET_QA_PATH, OUTPUTS_DIR, PROCESSED_DATA_DIR
from retriever.bm25_retriever import BM25Retriever
from utils.io_utils import load_json
from utils.text_utils import contains_any_answer


DEFAULT_DATASETS = ["squad", "triviaqa", "webq", "nq", "popqa"]


def find_processed_paths(dataset: str) -> tuple[Path, Path]:
    qa_path = PROCESSED_DATA_DIR / f"{dataset}_qa.json"
    corpus_path = PROCESSED_DATA_DIR / f"{dataset}_corpus.json"
    if qa_path.exists() and corpus_path.exists():
        return qa_path, corpus_path
    if dataset == "squad" and DATASET_QA_PATH.exists() and DATASET_CORPUS_PATH.exists():
        return DATASET_QA_PATH, DATASET_CORPUS_PATH
    return qa_path, corpus_path


def select_records(records: List[Dict], split: str, limit: int | None) -> List[Dict]:
    if split != "all":
        selected = [
            r for r in records
            if str(r.get("split", "")).lower() == split.lower()
        ]
    else:
        selected = list(records)

    if limit is not None and limit > 0:
        selected = selected[:limit]
    return selected


def extract_text(item: Dict) -> str:
    for key in ["text", "passage", "content", "body", "context"]:
        if key in item:
            return str(item.get(key) or "")
    return str(item)


def diagnose_dataset(
    dataset: str,
    split: str,
    top_ks: List[int],
    limit: int | None,
    output_rows: List[Dict],
) -> Dict:
    qa_path, corpus_path = find_processed_paths(dataset)
    if not qa_path.exists() or not corpus_path.exists():
        summary = {
            "dataset": dataset,
            "split": split,
            "count": 0,
            "answerable_count": 0,
            "corpus_count": 0,
            "status": "missing_processed_files",
        }
        for k in top_ks:
            summary[f"support_rate_at_{k}"] = 0.0
        return summary

    qa_records = select_records(load_json(qa_path), split=split, limit=limit)
    corpus = load_json(corpus_path)
    retriever = BM25Retriever(corpus)
    max_k = max(top_ks)

    support_counts = {k: 0 for k in top_ks}
    answerable_count = 0

    for idx, item in enumerate(qa_records):
        question = item.get("question", "")
        gold_answers = item.get("gold_answers", []) or []
        answerable = bool(gold_answers)
        if answerable:
            answerable_count += 1

        retrieved = retriever.retrieve(question, top_k=max_k, offset=0, exclude_ids=set())
        support_at_k = {}
        for k in top_ks:
            has_support = any(
                contains_any_answer(extract_text(doc), gold_answers)
                for doc in retrieved[:k]
            )
            support_at_k[k] = int(has_support)
            support_counts[k] += int(has_support)

        output_rows.append({
            "dataset": dataset,
            "split": item.get("split", split),
            "row_index": idx,
            "question_id": item.get("id", ""),
            "question": question,
            "gold_answers": " || ".join(map(str, gold_answers)),
            "answerable": int(answerable),
            **{f"support_at_{k}": support_at_k[k] for k in top_ks},
            "top_doc_ids": " || ".join(str(doc.get("id", "")) for doc in retrieved),
            "top_scores": " || ".join(f"{float(doc.get('score', 0.0)):.4f}" for doc in retrieved),
        })

    summary = {
        "dataset": dataset,
        "split": split,
        "count": len(qa_records),
        "answerable_count": answerable_count,
        "corpus_count": len(corpus),
        "status": "ok",
    }
    for k in top_ks:
        summary[f"support_rate_at_{k}"] = (
            support_counts[k] / len(qa_records) if qa_records else 0.0
        )
    return summary


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    parser.add_argument("--top-k", nargs="*", type=int, default=[1, 3, 5])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUTS_DIR / "diagnostics"),
    )
    args = parser.parse_args()

    top_ks = sorted(set(k for k in args.top_k if k > 0))
    if not top_ks:
        raise ValueError("--top-k must contain at least one positive integer")

    detail_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    for dataset in args.datasets:
        summary_rows.append(
            diagnose_dataset(
                dataset=dataset,
                split=args.split,
                top_ks=top_ks,
                limit=args.limit,
                output_rows=detail_rows,
            )
        )

    output_dir = Path(args.output_dir)
    detail_path = output_dir / "retrieval_support.csv"
    summary_path = output_dir / "retrieval_summary.csv"
    write_csv(detail_path, detail_rows)
    write_csv(summary_path, summary_rows)

    print(f"Saved retrieval support details to {detail_path}")
    print(f"Saved retrieval support summary to {summary_path}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
