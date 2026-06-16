import argparse
import json
from itertools import islice
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset

from config import DEFAULT_DATASET_DOWNLOAD_LIMIT, RAW_DATA_DIR


def save_jsonl(data_iter, path: Path, limit: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in islice(data_iter, limit):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_hf_stream(dataset_name: str, kwargs: Dict):
    kwargs = dict(kwargs)
    name = kwargs.pop("name", None)
    if name is None:
        return load_dataset(dataset_name, streaming=True, **kwargs)
    return load_dataset(dataset_name, name, streaming=True, **kwargs)


def download_with_fallbacks(
    dataset_name: str,
    candidates: List[Dict],
    out_path: Path,
    limit: int,
) -> None:
    errors = []
    for kwargs in candidates:
        try:
            print(f"[download] {dataset_name} {kwargs} -> {out_path}")
            ds = load_hf_stream(dataset_name, kwargs)
            count = save_jsonl(ds, out_path, limit=limit)
            if count > 0:
                print(f"[OK] saved {count} rows to {out_path}")
                return
            errors.append(f"{kwargs}: empty stream")
        except Exception as exc:
            errors.append(f"{kwargs}: {exc}")

    if out_path.exists():
        out_path.unlink()
    raise RuntimeError(
        f"Failed to download non-empty data for {dataset_name} -> {out_path}. "
        + " | ".join(errors)
    )


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

    datasets: List[Tuple[str, List[Dict], Path]] = [
        ("squad", [{"split": "train"}], save_dir / "squad_train.jsonl"),
        ("squad", [{"split": "validation"}], save_dir / "squad_dev.jsonl"),
        ("natural_questions", [{"split": "train"}], save_dir / "nq.jsonl"),
        (
            "trivia_qa",
            [
                {"name": "rc", "split": "train"},
                {"name": "unfiltered.nocontext", "split": "train"},
                {"name": "unfiltered", "split": "train"},
            ],
            save_dir / "trivia_train.jsonl",
        ),
        (
            "trivia_qa",
            [
                {"name": "rc", "split": "validation"},
                {"name": "unfiltered.nocontext", "split": "validation"},
                {"name": "unfiltered", "split": "validation"},
            ],
            save_dir / "trivia_dev.jsonl",
        ),
        ("web_questions", [{"split": "train"}], save_dir / "webq_train.jsonl"),
        ("web_questions", [{"split": "test"}], save_dir / "webq_test.jsonl"),
        ("akariasai/PopQA", [{"split": "test"}], save_dir / "popqa_test.jsonl"),
    ]

    for dataset_name, candidates, out_path in datasets:
        download_with_fallbacks(dataset_name, candidates, out_path, limit=limit)

    print("Done.")


if __name__ == "__main__":
    main()
