import argparse
from pathlib import Path
from typing import Any, Dict, List

from config import (
    MINI_DATASET_PATH,
    CORPUS_PATH,
    UTILITY_DATASET_PATH,
    GENERATOR_MODEL_NAME,
    MAX_INPUT_LENGTH,
    MAX_NEW_TOKENS,
    UTILITY_INCLUDE_SUPPORT_SCORE,
    UTILITY_POSITIVE_THRESHOLD,
)
from generator.qa_generator import QAGenerator
from retriever.bm25_retriever import BM25Retriever
from utils.io_utils import load_json, save_json
from utils.text_utils import contains_any_answer, qa_match


def compute_utility_score(
    pred_answer: str,
    gold_answers: List[str],
    passage: str,
    include_support: bool = True,
) -> Dict[str, Any]:
    answer_correct = qa_match(pred_answer, gold_answers)
    support = int(contains_any_answer(passage, gold_answers))

    utility_score = float(answer_correct)
    label = int(answer_correct)

    return {
        "answer_correct": int(answer_correct),
        "support": int(support),
        "utility_score": float(utility_score),
        "label": int(label),
    }


def extract_passage_text(p: Any) -> str:
    if isinstance(p, str):
        return p
    if isinstance(p, dict):
        for key in ["text", "passage", "content", "body", "context"]:
            if key in p:
                return str(p[key])
    return str(p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--output", type=str, default=str(UTILITY_DATASET_PATH))
    args = parser.parse_args()

    dataset = load_json(MINI_DATASET_PATH)
    corpus = load_json(CORPUS_PATH)

    if args.max_questions is not None:
        dataset = dataset[: args.max_questions]

    print(f"Loaded {len(dataset)} QA samples from {MINI_DATASET_PATH}")
    print(f"Loaded {len(corpus)} corpus passages from {CORPUS_PATH}")

    retriever = BM25Retriever(corpus)
    generator = QAGenerator(
        model_name=GENERATOR_MODEL_NAME,
        max_input_length=MAX_INPUT_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    utility_dataset = []
    total_pairs = 0
    positive_count = 0
    output_path = Path(args.output)

    for sample_idx, sample in enumerate(dataset):
        question = sample["question"]
        gold_answers = sample["gold_answers"]
        qid = sample.get("id", f"q_{sample_idx}")

        retrieved = retriever.retrieve(question, top_k=args.top_k)

        for passage_idx, item in enumerate(retrieved):
            passage_text = extract_passage_text(item)
            pred_answer = generator.answer_with_single_passage(question, passage_text)

            stats = compute_utility_score(
                pred_answer=pred_answer,
                gold_answers=gold_answers,
                passage=passage_text,
                include_support=UTILITY_INCLUDE_SUPPORT_SCORE,
            )

            record = {
                "question_id": qid,
                "question": question,
                "gold_answers": gold_answers,
                "passage_id": item.get("id", f"{qid}_p_{passage_idx}") if isinstance(item, dict) else f"{qid}_p_{passage_idx}",
                "passage_index": passage_idx,
                "passage_rank": passage_idx + 1,
                "bm25_score": float(item.get("score", 0.0)) if isinstance(item, dict) else 0.0,
                "passage": passage_text,
                "pred_answer": pred_answer,
                "pred_answer_in_passage": int(pred_answer.lower() in passage_text.lower()) if pred_answer else 0,
                "answer_correct": stats["answer_correct"],
                "support": stats["support"],
                "utility_score": stats["utility_score"],
                "label": stats["label"],
            }

            utility_dataset.append(record)
            total_pairs += 1
            positive_count += stats["label"]

        if (sample_idx + 1) % 10 == 0 or sample_idx == len(dataset) - 1:
            print(
                f"Processed {sample_idx + 1}/{len(dataset)} questions | "
                f"Pairs so far: {total_pairs}"
            )

        if (sample_idx + 1) % args.save_every == 0:
            save_json(utility_dataset, output_path)
            print(f"[Checkpoint] Saved {len(utility_dataset)} records -> {output_path}")

    save_json(utility_dataset, output_path)

    pos_ratio = positive_count / max(total_pairs, 1)
    print(f"\nSaved {len(utility_dataset)} utility samples to {output_path}")
    print(f"Positive labels: {positive_count}/{total_pairs} ({pos_ratio:.2%})")


if __name__ == "__main__":
    main()