import argparse
import json
from itertools import islice
from pathlib import Path

from datasets import load_dataset

from config import DEFAULT_DATASET_DOWNLOAD_LIMIT, RAW_DATA_DIR


def save_jsonl(data_iter, path: Path, limit: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in islice(data_iter, limit):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_DATASET_DOWNLOAD_LIMIT,
        help="Maximum examples per split for the TPAMI prototype.",
    )
    args = parser.parse_args()

    limit = max(1, int(args.limit))
    save_dir = RAW_DATA_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading streaming QA subsets with limit={limit} per split...")

    datasets = [
        ("squad", {"split": "train"}, save_dir / "squad_train.jsonl"),
        ("squad", {"split": "validation"}, save_dir / "squad_dev.jsonl"),
        ("natural_questions", {"split": "train"}, save_dir / "nq.jsonl"),
        ("trivia_qa", {"name": "rc", "split": "train"}, save_dir / "trivia_train.jsonl"),
        ("trivia_qa", {"name": "rc", "split": "validation"}, save_dir / "trivia_dev.jsonl"),
        ("web_questions", {"split": "train"}, save_dir / "webq_train.jsonl"),
        ("web_questions", {"split": "test"}, save_dir / "webq_test.jsonl"),
        ("akariasai/PopQA", {"split": "test"}, save_dir / "popqa_test.jsonl"),
    ]

    for dataset_name, kwargs, out_path in datasets:
        name = kwargs.pop("name", None)
        print(f"[download] {dataset_name} -> {out_path}")
        if name is None:
            ds = load_dataset(dataset_name, streaming=True, **kwargs)
        else:
            ds = load_dataset(dataset_name, name, streaming=True, **kwargs)
        save_jsonl(ds, out_path, limit=limit)

    print("Done.")


if __name__ == "__main__":
    main()
