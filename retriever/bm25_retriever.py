import json
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, corpus_path: str):
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        self.texts = [doc["text"] for doc in self.corpus]
        self.ids = [doc["id"] for doc in self.corpus]
        self.tokenized_corpus = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 3):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.ids, self.texts, scores),
            key=lambda x: x[2],
            reverse=True
        )[:top_k]

        return [
            {"doc_id": doc_id, "text": text, "score": float(score)}
            for doc_id, text, score in ranked
        ]