from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Set
import re

from rank_bm25 import BM25Okapi

from utils.io_utils import load_json


class BM25Retriever:
    def __init__(self, corpus_source: Union[str, Path, List[Dict[str, Any]]]):
        if isinstance(corpus_source, (str, Path)):
            self.corpus = load_json(corpus_source)
        elif isinstance(corpus_source, list):
            self.corpus = corpus_source
        else:
            raise TypeError(
                "corpus_source must be a file path or a list of corpus records, "
                f"got {type(corpus_source)}"
            )

        if not isinstance(self.corpus, list):
            raise ValueError("Corpus must be a list of passages/records.")

        self.texts = [self._extract_text(doc) for doc in self.corpus]
        self.tokenized_corpus = [self._tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _extract_text(self, doc: Any) -> str:
        if isinstance(doc, str):
            return doc

        if isinstance(doc, dict):
            for key in ["text", "passage", "content", "body", "context"]:
                if key in doc and doc[key]:
                    return str(doc[key])

        return str(doc)

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        return re.findall(r"\w+", text)

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        offset: int = 0,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not question or not str(question).strip():
            return []

        exclude_ids = exclude_ids or set()

        tokenized_query = self._tokenize(question)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []
        skipped = 0

        for idx in ranked_indices:
            doc = self.corpus[idx]

            if isinstance(doc, dict):
                result = dict(doc)
            else:
                result = {
                    "id": f"doc_{idx}",
                    "text": str(doc),
                }

            doc_id = result.get("id", f"doc_{idx}")
            if doc_id in exclude_ids:
                continue

            if skipped < offset:
                skipped += 1
                continue

            result["id"] = doc_id
            result["score"] = float(scores[idx])
            results.append(result)

            if len(results) >= top_k:
                break

        return results