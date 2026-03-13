import argparse
import hashlib
from typing import Any, Dict, List, Tuple

from config import (
    DATASET_QA_PATH,
    DATASET_CORPUS_PATH,
    DATASET_STATS_PATH,
    SQUAD_TRAIN_V11_PATH,
    SQUAD_DEV_V11_PATH,
    SQUAD_TRAIN_V20_PATH,
    SQUAD_DEV_V20_PATH,
    SQUAD_INCLUDE_TITLE_IN_PASSAGE,
    SQUAD_MAX_CONTEXT_CHARS,
)
from utils.io_utils import load_json, save_json


def normalize_answers(answers: List[str]) -> List[str]:
    cleaned = []
    seen = set()
    for ans in answers:
        if ans is None:
            continue
        x = str(ans).strip()
        if not x:
            continue
        if x not in seen:
            cleaned.append(x)
            seen.add(x)
    return cleaned


def build_passage_text(title: str, context: str) -> str:
    context = context.strip()
    if SQUAD_MAX_CONTEXT_CHARS is not None:
        context = context[:SQUAD_MAX_CONTEXT_CHARS]

    if SQUAD_INCLUDE_TITLE_IN_PASSAGE and title:
        return f"Title: {title}\n{context}"
    return context


def make_context_id(title: str, context: str) -> str:
    digest = hashlib.md5(f"{title}||{context}".encode("utf-8")).hexdigest()[:16]
    return f"squad_ctx_{digest}"


def parse_squad_file(
    raw_data: Dict[str, Any],
    split_name: str,
    version: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    qa_records: List[Dict[str, Any]] = []
    corpus_records: List[Dict[str, Any]] = []

    seen_context_ids = set()

    articles = raw_data.get("data", [])
    for article_idx, article in enumerate(articles):
        title = article.get("title", f"untitled_{article_idx}")
        paragraphs = article.get("paragraphs", [])

        for para_idx, para in enumerate(paragraphs):
            context = para.get("context", "").strip()
            if not context:
                continue

            context_id = make_context_id(title, context)

            if context_id not in seen_context_ids:
                corpus_records.append({
                    "id": context_id,
                    "dataset": "squad",
                    "version": version,
                    "source_split": split_name,
                    "title": title,
                    "paragraph_index": para_idx,
                    "text": build_passage_text(title, context),
                    "context": context,
                })
                seen_context_ids.add(context_id)

            qas = para.get("qas", [])
            for qa_idx, qa in enumerate(qas):
                qid = qa.get("id", f"{split_name}_{article_idx}_{para_idx}_{qa_idx}")
                question = str(qa.get("question", "")).strip()
                is_impossible = bool(qa.get("is_impossible", False))

                answers_raw = qa.get("answers", [])
                gold_answers = normalize_answers([a.get("text", "") for a in answers_raw])

                # SQuAD 2.0 中不可回答题可被过滤或保留
                qa_records.append({
                    "id": qid,
                    "dataset": "squad",
                    "version": version,
                    "split": split_name,
                    "question": question,
                    "gold_answers": gold_answers,
                    "metadata": {
                        "title": title,
                        "context_id": context_id,
                        "is_impossible": is_impossible,
                        "source_answer_count": len(answers_raw),
                    }
                })

    return qa_records, corpus_records


def get_input_paths(version: str) -> List[Tuple[str, Any]]:
    if version == "v1.1":
        return [
            ("train", SQUAD_TRAIN_V11_PATH),
            ("dev", SQUAD_DEV_V11_PATH),
        ]
    elif version == "v2.0":
        return [
            ("train", SQUAD_TRAIN_V20_PATH),
            ("dev", SQUAD_DEV_V20_PATH),
        ]
    else:
        raise ValueError(f"Unsupported version: {version}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v1.1", choices=["v1.1", "v2.0"])
    parser.add_argument(
        "--keep-impossible",
        action="store_true",
        help="For SQuAD 2.0, keep impossible questions instead of filtering them out."
    )
    args = parser.parse_args()

    all_qa: List[Dict[str, Any]] = []
    corpus_map: Dict[str, Dict[str, Any]] = {}

    input_paths = get_input_paths(args.version)

    for split_name, path in input_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing raw SQuAD file: {path}\n"
                f"Please download it first."
            )

        raw_data = load_json(path)
        qa_records, corpus_records = parse_squad_file(
            raw_data=raw_data,
            split_name=split_name,
            version=args.version,
        )

        if args.version == "v2.0" and not args.keep_impossible:
            qa_records = [
                x for x in qa_records
                if not x["metadata"].get("is_impossible", False)
            ]

        all_qa.extend(qa_records)

        for c in corpus_records:
            corpus_map[c["id"]] = c

    all_corpus = list(corpus_map.values())

    stats = {
        "dataset": "squad",
        "version": args.version,
        "qa_count": len(all_qa),
        "corpus_count": len(all_corpus),
        "train_count": sum(1 for x in all_qa if x["split"] == "train"),
        "dev_count": sum(1 for x in all_qa if x["split"] == "dev"),
        "impossible_count": sum(
            1 for x in all_qa if x["metadata"].get("is_impossible", False)
        ),
    }

    save_json(all_qa, DATASET_QA_PATH)
    save_json(all_corpus, DATASET_CORPUS_PATH)
    save_json(stats, DATASET_STATS_PATH)

    print(f"Saved QA records    : {len(all_qa)} -> {DATASET_QA_PATH}")
    print(f"Saved corpus records: {len(all_corpus)} -> {DATASET_CORPUS_PATH}")
    print(f"Saved stats         : {DATASET_STATS_PATH}")
    print(stats)


if __name__ == "__main__":
    main()