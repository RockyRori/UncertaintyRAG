from rank_bm25 import BM25Okapi
from utils.io_utils import load_json
from utils.text_utils import normalize_text


class BM25Retriever:
    def __init__(self, corpus_path: str):
        self.corpus = load_json(corpus_path)
        self.passages = [item["text"] for item in self.corpus]
        self.ids = [item["id"] for item in self.corpus]

        self.tokenized_corpus = [
            normalize_text(p).split() for p in self.passages
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, question: str, top_k: int = 5, offset: int = 0, exclude_ids: set | None = None):
        query_tokens = normalize_text(question).split()
        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(self.ids, self.passages, scores),
            key=lambda x: x[2],
            reverse=True
        )

        exclude_ids = exclude_ids or set()
        filtered = [
            (pid, text, score)
            for pid, text, score in ranked
            if pid not in exclude_ids
        ]

        sliced = filtered[offset: offset + top_k]

        return [
            {
                "id": pid,
                "text": text,
                "score": float(score)
            }
            for pid, text, score in sliced
        ]