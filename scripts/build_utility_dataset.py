from config import MINI_DATASET_PATH, UTILITY_DATASET_PATH
from utils.io_utils import load_json, save_json
from utils.text_utils import contains_any_answer


def main() -> None:
    dataset = load_json(MINI_DATASET_PATH)
    utility_dataset = []

    if not dataset:
        raise ValueError("mini_dataset.json is empty.")

    print("Example sample keys:", list(dataset[0].keys()))

    for sample in dataset:
        question = sample["question"]
        passages = sample["passages"]
        gold_answers = sample["gold_answers"]

        for passage in passages:
            label = 1 if contains_any_answer(passage, gold_answers) else 0
            utility_dataset.append({
                "question": question,
                "passage": passage,
                "label": label
            })

    save_json(utility_dataset, UTILITY_DATASET_PATH)
    print(f"Saved {len(utility_dataset)} utility samples to {UTILITY_DATASET_PATH}")


if __name__ == "__main__":
    main()