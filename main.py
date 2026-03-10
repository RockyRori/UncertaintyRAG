import json
from retriever.bm25_retriever import BM25Retriever
from generator.qa_generator import QAGenerator
from uncertainty.scorer import UncertaintyScorer
from evaluation.metrics import exact_match


def main():
    corpus_path = "data/demo/corpus.json"
    qa_path = "data/demo/qa.json"

    retriever = BM25Retriever(corpus_path)
    generator = QAGenerator(model_name="google/flan-t5-base")
    scorer = UncertaintyScorer()

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    all_results = []

    for item in qa_data:
        question = item["question"]
        gold_answer = item["answer"]

        retrieved = retriever.retrieve(question, top_k=3)
        contexts = [doc["text"] for doc in retrieved]

        pred_answer = generator.generate(question, contexts)
        uncertainty = scorer.retrieval_uncertainty(retrieved)
        em = exact_match(pred_answer, gold_answer)

        result = {
            "question": question,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "retrieved_docs": retrieved,
            "uncertainty": uncertainty,
            "exact_match": em
        }

        all_results.append(result)

        print("=" * 80)
        print("Question:", question)
        print("Gold:", gold_answer)
        print("Pred:", pred_answer)
        print("Uncertainty:", uncertainty)
        print("EM:", em)

    with open("outputs/demo_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()