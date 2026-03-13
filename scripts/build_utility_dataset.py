from typing import Any, Dict, List

from config import (
    MINI_DATASET_PATH,
    UTILITY_DATASET_PATH,
    GENERATOR_MODEL_NAME,
    MAX_INPUT_LENGTH,
    MAX_NEW_TOKENS,
    UTILITY_INCLUDE_SUPPORT_SCORE,
    UTILITY_POSITIVE_THRESHOLD,
)
from generator.qa_generator import QAGenerator
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

    if include_support:
        utility_score = 0.5 * answer_correct + 0.5 * support
    else:
        utility_score = float(answer_correct)

    label = int(utility_score >= UTILITY_POSITIVE_THRESHOLD)

    return {
        "answer_correct": int(answer_correct),
        "support": int(support),
        "utility_score": float(utility_score),
        "label": int(label),
    }


def main() -> None:
    dataset = load_json(MINI_DATASET_PATH)
    utility_dataset = []

    if not dataset:
        raise ValueError("mini_dataset.json is empty.")

    print(f"Loaded {len(dataset)} QA samples from {MINI_DATASET_PATH}")

    generator = QAGenerator(
        model_name=GENERATOR_MODEL_NAME,
        max_input_length=MAX_INPUT_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    total_pairs = 0
    positive_count = 0

    for sample_idx, sample in enumerate(dataset):
        question = sample["question"]
        passages = sample["passages"]
        gold_answers = sample["gold_answers"]
        qid = sample.get("id", f"q_{sample_idx}")

        for passage_idx, passage in enumerate(passages):
            pred_answer = generator.answer_with_single_passage(question, passage)

            stats = compute_utility_score(
                pred_answer=pred_answer,
                gold_answers=gold_answers,
                passage=passage,
                include_support=UTILITY_INCLUDE_SUPPORT_SCORE,
            )

            record = {
                "question_id": qid,
                "question": question,
                "gold_answers": gold_answers,
                "passage_id": f"{qid}_p_{passage_idx}",
                "passage_index": passage_idx,
                "passage": passage,
                "pred_answer": pred_answer,
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

    save_json(utility_dataset, UTILITY_DATASET_PATH)

    pos_ratio = positive_count / max(total_pairs, 1)

    print(f"\nSaved {len(utility_dataset)} utility samples to {UTILITY_DATASET_PATH}")
    print(f"Positive labels: {positive_count}/{total_pairs} ({pos_ratio:.2%})")

    if utility_dataset:
        print("\nExample utility record:")
        example = utility_dataset[0]
        for k, v in example.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()