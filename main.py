"""
Phase 2 main entry.
Phase 3 please use: python main_phase3.py
"""
from config import MINI_DATASET_PATH
from utils.io_utils import load_json
from inference.predict_utility import UtilityPredictor


def dummy_answer(question: str, passages: list[str], gold_answers: list[str]) -> str:
    # 这里先保留一个简单占位
    # 如果你之前已有 phase2A 的 answer generator，就把这个换成你原来的逻辑
    return gold_answers[0]


def exact_match(pred: str, gold_answers: list[str]) -> int:
    pred = pred.strip().lower()
    return int(any(pred == g.strip().lower() for g in gold_answers))


def main():
    dataset = load_json(MINI_DATASET_PATH)
    predictor = UtilityPredictor()

    for sample in dataset[:10]:
        question = sample["question"]
        passages = sample["passages"]
        gold_answers = sample["gold_answers"]

        pred = dummy_answer(question, passages, gold_answers)
        em = exact_match(pred, gold_answers)

        utilities = predictor.predict_batch(question, passages)
        confidence = max(utilities)
        uncertainty = 1.0 - confidence

        print("=" * 80)
        print(f"Question: {question}")
        print(f"Gold: {gold_answers}")
        print(f"Pred: {pred}")
        print(f"Utilities: {[round(u, 4) for u in utilities]}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Uncertainty: {uncertainty:.4f}")
        print(f"EM: {em}")


if __name__ == "__main__":
    main()