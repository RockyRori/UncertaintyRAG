import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack

from config import DROPOUT, HIDDEN_DIM, TFIDF_VECTORIZER_PATH, UTILITY_MODEL_PATH
from models.utility_predictor import UtilityMLP
from utils.feature_utils import (
    build_text_feature_from_values,
    structured_features_from_values,
)


class UtilityPredictor:
    def __init__(
        self,
        model_path: str | Path | None = None,
        vectorizer_path: str | Path | None = None,
    ):
        self.model_path = Path(model_path) if model_path is not None else UTILITY_MODEL_PATH
        self.vectorizer_path = Path(vectorizer_path) if vectorizer_path is not None else TFIDF_VECTORIZER_PATH

        with open(self.vectorizer_path, "rb") as f:
            bundle = pickle.load(f)

        if isinstance(bundle, dict):
            self.vectorizer = bundle["vectorizer"]
            self.scaler = bundle.get("scaler", None)
            self.structured_feature_names = bundle.get("structured_feature_names", [])
        else:
            self.vectorizer = bundle
            self.scaler = None
            self.structured_feature_names = []

        text_dim = len(self.vectorizer.get_feature_names_out())
        struct_dim = len(self.structured_feature_names)
        input_dim = text_dim + struct_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UtilityMLP(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
        ).to(self.device)

        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    @staticmethod
    def build_text_feature(question: str, pred_answer: str, passage: str) -> str:
        return build_text_feature_from_values(question, pred_answer, passage)

    def extract_structured_features(
        self,
        question: str,
        passage: str,
        pred_answer: str,
        bm25_score: float = 0.0,
        passage_rank: int = 0,
    ) -> List[float]:
        return structured_features_from_values(
            question=question,
            passage=passage,
            pred_answer=pred_answer,
            bm25_score=bm25_score,
            passage_rank=passage_rank,
            feature_names=self.structured_feature_names,
        )

    def _build_input_vector(
        self,
        question: str,
        passage: str,
        pred_answer: str,
        bm25_score: float = 0.0,
        passage_rank: int = 0,
    ) -> np.ndarray:
        text = self.build_text_feature(question, pred_answer, passage)
        x_text = self.vectorizer.transform([text])

        if self.scaler is not None and self.structured_feature_names:
            x_struct = np.array(
                [
                    self.extract_structured_features(
                        question=question,
                        passage=passage,
                        pred_answer=pred_answer,
                        bm25_score=bm25_score,
                        passage_rank=passage_rank,
                    )
                ],
                dtype=np.float32,
            )
            x_struct = self.scaler.transform(x_struct)
            x_struct_sparse = csr_matrix(x_struct)
            x_all = hstack([x_text, x_struct_sparse]).astype(np.float32)
            return x_all.toarray()

        return x_text.toarray().astype(np.float32)

    def predict_one(
        self,
        question: str,
        passage: str,
        pred_answer: str,
        bm25_score: float = 0.0,
        passage_rank: int = 0,
    ) -> float:
        x = self._build_input_vector(
            question=question,
            passage=passage,
            pred_answer=pred_answer,
            bm25_score=bm25_score,
            passage_rank=passage_rank,
        )
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(x_tensor)
            prob = torch.sigmoid(logits).item()

        # Keep the old mild calibration transform for compatibility. A proper
        # calibration head is a TPAMI follow-up, not a hidden feature hack.
        return float(prob ** 0.5)

    def predict_batch(
        self,
        question: str,
        passages: List[str],
        pred_answers: List[str] | None = None,
        bm25_scores: List[float] | None = None,
        passage_ranks: List[int] | None = None,
    ) -> List[float]:
        if pred_answers is None:
            pred_answers = [""] * len(passages)
        if bm25_scores is None:
            bm25_scores = [0.0] * len(passages)
        if passage_ranks is None:
            passage_ranks = list(range(1, len(passages) + 1))

        return [
            self.predict_one(
                question=question,
                passage=passage,
                pred_answer=pred_answer,
                bm25_score=bm25_score,
                passage_rank=rank,
            )
            for passage, pred_answer, bm25_score, rank in zip(
                passages, pred_answers, bm25_scores, passage_ranks
            )
        ]
