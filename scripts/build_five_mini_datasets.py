import json
import random
from pathlib import Path
from typing import List, Dict, Any

RANDOM_SEED = 42
TRAIN_SIZE = 50
TEST_SIZE = 50

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/mini_datasets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["squad", "nq", "triviaqa", "webq", "popqa"]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def has_valid_answer(item: Dict[str, Any]) -> bool:
    answers = item.get("gold_answers", [])
    return isinstance(answers, list) and len([x for x in answers if str(x).strip()]) > 0


def normalize_item(item: Dict[str, Any], dataset_name: str, idx: int) -> Dict[str, Any]:
    qid = item.get("id", f"{dataset_name}_{idx}")
    question = str(item.get("question", "")).strip()
    split = str(item.get("split", "")).strip().lower()
    gold_answers = item.get("gold_answers", [])
    if not isinstance(gold_answers, list):
        gold_answers = [str(gold_answers)]

    gold_answers = [str(x).strip() for x in gold_answers if str(x).strip()]

    return {
        "id": str(qid),
        "dataset": dataset_name,
        "split": split,
        "question": question,
        "gold_answers": gold_answers
    }


def clean_qa_items(items: List[Dict[str, Any]], dataset_name: str) -> List[Dict[str, Any]]:
    cleaned = []
    seen = set()

    for idx, item in enumerate(items):
        x = normalize_item(item, dataset_name, idx)

        if not x["question"]:
            continue
        if not has_valid_answer(x):
            continue

        key = (x["question"], tuple(x["gold_answers"]))
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(x)

    return cleaned


def split_by_existing_labels(items: List[Dict[str, Any]]):
    train_items = [x for x in items if x.get("split") == "train"]
    test_items = [x for x in items if x.get("split") in {"test", "dev", "validation"}]
    other_items = [x for x in items if x.get("split") not in {"train", "test", "dev", "validation"}]
    return train_items, test_items, other_items


def sample_items(items: List[Dict[str, Any]], k: int, rng: random.Random) -> List[Dict[str, Any]]:
    if len(items) <= k:
        out = items[:]
        rng.shuffle(out)
        return out
    return rng.sample(items, k)


def assign_split(items: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    out = []
    for x in items:
        y = dict(x)
        y["split"] = split_name
        out.append(y)
    return out


def build_mini_dataset(dataset_name: str, train_size: int = TRAIN_SIZE, test_size: int = TEST_SIZE):
    qa_path = PROCESSED_DIR / f"{dataset_name}_qa.json"
    corpus_path = PROCESSED_DIR / f"{dataset_name}_corpus.json"

    if not qa_path.exists():
        print(f"[WARN] Missing {qa_path}, skip {dataset_name}.")
        return None

    qa_items = load_json(qa_path)
    qa_items = clean_qa_items(qa_items, dataset_name)

    if len(qa_items) == 0:
        print(f"[WARN] No valid QA items in {qa_path}, skip {dataset_name}.")
        return None

    corpus_items = load_json(corpus_path) if corpus_path.exists() else []

    rng = random.Random(RANDOM_SEED)

    train_existing, test_existing, other_items = split_by_existing_labels(qa_items)

    # 优先使用已有 split
    train_pool = train_existing[:]
    test_pool = test_existing[:]

    # 如果没有显式 split，或者样本不足，从 other 里补
    remaining_other = other_items[:]
    rng.shuffle(remaining_other)

    if len(train_pool) < train_size:
        need = train_size - len(train_pool)
        train_pool.extend(remaining_other[:need])
        remaining_other = remaining_other[need:]

    if len(test_pool) < test_size:
        need = test_size - len(test_pool)
        test_pool.extend(remaining_other[:need])
        remaining_other = remaining_other[need:]

    # 如果仍不足，从全体里重新切
    total_needed = train_size + test_size
    if len(train_pool) < train_size or len(test_pool) < test_size:
        print(f"[INFO] {dataset_name}: existing splits insufficient, fallback to random split.")

        shuffled = qa_items[:]
        rng.shuffle(shuffled)

        if len(shuffled) < total_needed:
            # 数据太少就尽量切，至少保证 train/test 都有
            actual_train = min(train_size, max(1, len(shuffled) // 2))
            actual_test = min(test_size, max(1, len(shuffled) - actual_train))
        else:
            actual_train = train_size
            actual_test = test_size

        train_items = assign_split(shuffled[:actual_train], "train")
        test_items = assign_split(shuffled[actual_train:actual_train + actual_test], "test")
    else:
        train_items = assign_split(sample_items(train_pool, train_size, rng), "train")
        used_ids = set(x["id"] for x in train_items)

        filtered_test_pool = [x for x in test_pool if x["id"] not in used_ids]
        if len(filtered_test_pool) < test_size:
            # 不够就从原 test_pool 里硬取，避免空转
            filtered_test_pool = test_pool

        test_items = assign_split(sample_items(filtered_test_pool, test_size, rng), "test")

    mini_qa = train_items + test_items

    dataset_out_dir = OUTPUT_DIR / dataset_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    save_json(mini_qa, dataset_out_dir / f"{dataset_name}_mini_qa.json")
    save_json(corpus_items, dataset_out_dir / f"{dataset_name}_mini_corpus.json")

    stats = {
        "dataset": dataset_name,
        "original_qa_count": len(qa_items),
        "original_corpus_count": len(corpus_items),
        "mini_train_count": len([x for x in mini_qa if x["split"] == "train"]),
        "mini_test_count": len([x for x in mini_qa if x["split"] == "test"]),
        "output_qa_path": str(dataset_out_dir / f"{dataset_name}_mini_qa.json"),
        "output_corpus_path": str(dataset_out_dir / f"{dataset_name}_mini_corpus.json")
    }

    save_json(stats, dataset_out_dir / f"{dataset_name}_mini_stats.json")
    print(
        f"[OK] {dataset_name}: "
        f"train={stats['mini_train_count']}, "
        f"test={stats['mini_test_count']}, "
        f"corpus={stats['original_corpus_count']}"
    )
    return stats


def main():
    all_stats = {}

    for dataset_name in DATASETS:
        stats = build_mini_dataset(dataset_name)
        if stats is not None:
            all_stats[dataset_name] = stats

    save_json(all_stats, OUTPUT_DIR / "mini_stats.json")
    print(f"\n[OK] Saved summary to {OUTPUT_DIR / 'mini_stats.json'}")
    print("[DONE] All mini datasets are ready.")


if __name__ == "__main__":
    main()