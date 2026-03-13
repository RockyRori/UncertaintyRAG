from pathlib import Path
from typing import Any, Dict, List, Union
import re

from rank_bm25 import BM25Okapi

from utils.io_utils import load_json


class BM25Retriever:
    def __init__(self, corpus_source: Union[str, Path, List[Dict[str, Any]]]):
        """
        corpus_source can be:
        1. a file path to a json corpus
        2. an already loaded list of corpus records

        Each corpus record is expected to be a dict like:
        {
            "id": "...",
            "text": "...",
            ...
        }
        """
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

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query or not str(query).strip():
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in ranked_indices:
            doc = self.corpus[idx]

            if isinstance(doc, dict):
                result = dict(doc)
            else:
                result = {
                    "id": f"doc_{idx}",
                    "text": str(doc),
                }

            result["score"] = float(scores[idx])
            results.append(result)

        return results