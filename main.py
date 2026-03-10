import torch
from tqdm import tqdm

from config import Config
from utils.io_utils import load_json, save_json, save_jsonl
from retriever.bm25_retriever import BM25Retriever
from generator.qa_generator import QAGenerator
from uncertainty.scorer import WeakUtilityScorer
from evaluation.metrics import (
    compute_accuracy,
    compute_auroc,
    compute_avg_uncertainty,
    selective_accuracy
)
from utils.text_utils import qa_match


def main():
    cfg = Config()

    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Device set to use {device}")

    dataset = load_json(cfg.dataset_path)

    retriever = BM25Retriever(cfg.corpus_path)
    generator = QAGenerator(
        model_name=cfg.model_name,
        device=device,
        max_input_length=cfg.max_input_length,
        max_new_tokens=cfg.max_new_tokens
    )
    scorer = WeakUtilityScorer()

    records = []
    error_cases = []

    for sample in tqdm(dataset, desc="Running Phase 2A"):
        qid = sample["id"]
        question = sample["question"]
        gold_answers = sample["gold_answers"]

        # 1) retrieve top-k passages
        retrieved = retriever.retrieve(question, top_k=cfg.top_k)
        retrieved_texts = [x["text"] for x in retrieved]

        # 2) final answer with all passages
        final_answer = generator.answer_with_passages(question, retrieved_texts)
        final_correct = qa_match(final_answer, gold_answers)

        # 3) single-passage answers -> pseudo utilities
        single_passage_answers = []
        utilities = []

        for p in retrieved_texts:
            sp_answer = generator.answer_with_single_passage(question, p)
            utility = scorer.score_single_passage_utility(sp_answer, gold_answers)

            single_passage_answers.append(sp_answer)
            utilities.append(utility)

        confidence = scorer.aggregate_confidence(utilities)
        uncertainty = scorer.aggregate_uncertainty(utilities)

        row = {
            "id": qid,
            "question": question,
            "gold_answers": gold_answers,
            "retrieved_passages": retrieved,
            "final_answer": final_answer,
            "correct": final_correct,
            "single_passage_answers": single_passage_answers,
            "utilities": utilities,
            "confidence": confidence,
            "uncertainty": uncertainty,
        }
        records.append(row)

        if final_correct == 0:
            error_cases.append(row)

        print("=" * 80)
        print(f"Question: {question}")
        print(f"Gold: {gold_answers}")
        print(f"Pred: {final_answer}")
        print(f"Utilities: {utilities}")
        print(f"Confidence: {confidence}")
        print(f"Uncertainty: {uncertainty}")
        print(f"EM: {final_correct}")

    # metrics
    accuracy = compute_accuracy(records)
    auroc = compute_auroc(records)
    avg_uncertainty = compute_avg_uncertainty(records)
    selective_acc_80 = selective_accuracy(records, keep_ratio=0.8)

    metrics = {
        "num_samples": len(records),
        "accuracy": accuracy,
        "auroc": auroc,
        "avg_uncertainty": avg_uncertainty,
        "selective_accuracy_at_80": selective_acc_80
    }

    save_jsonl(records, cfg.predictions_path)
    save_json(metrics, cfg.metrics_path)
    save_json(error_cases, cfg.error_cases_path)

    print("\n" + "=" * 80)
    print("Phase 2A finished.")
    print(metrics)


if __name__ == "__main__":
    main()