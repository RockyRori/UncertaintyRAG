import pickle
import re
from typing import Dict, List

import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack

from config import TFIDF_VECTORIZER_PATH, UTILITY_MODEL_PATH, HIDDEN_DIM, DROPOUT
from models.utility_predictor import UtilityMLP


class UtilityPredictor:
    def __init__(self):
        with open(TFIDF_VECTORIZER_PATH, "rb") as f:
            bundle = pickle.load(f)

        # 兼容两种格式：
        # 1) 老版本：直接保存 vectorizer
        # 2) 新版本：保存 bundle dict
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
            dropout=DROPOUT
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(UTILITY_MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

    @staticmethod
    def safe_text(x) -> str:
        return "" if x is None else str(x)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", str(text).lower())

    @classmethod
    def overlap_ratio(cls, a: str, b: str) -> float:
        a_tokens = set(cls.tokenize(a))
        b_tokens = set(cls.tokenize(b))
        if not a_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / max(len(a_tokens), 1)

    @classmethod
    def answer_in_passage(cls, pred_answer: str, passage: str) -> int:
        pred = cls.safe_text(pred_answer).strip().lower()
        passage_low = cls.safe_text(passage).lower()
        if not pred:
            return 0
        return int(pred in passage_low)

    @staticmethod
    def build_text_feature(question: str, pred_answer: str, passage: str) -> str:
        question = "" if question is None else str(question)
        pred_answer = "" if pred_answer is None else str(pred_answer)
        passage = "" if passage is None else str(passage)

        return (
            f"question: {question} "
            f"[SEP] predicted_answer: {pred_answer} "
            f"[SEP] passage: {passage}"
        )

    def extract_structured_features(
        self,
        question: str,
        passage: str,
        pred_answer: str,
        bm25_score: float = 0.0,
        passage_rank: int = 0,
    ) -> List[float]:
        question = self.safe_text(question)
        passage = self.safe_text(passage)
        pred_answer = self.safe_text(pred_answer)

        q_tokens = self.tokenize(question)
        p_tokens = self.tokenize(passage)
        a_tokens = self.tokenize(pred_answer)

        question_len = float(len(q_tokens))
        passage_len = float(len(p_tokens))
        pred_answer_len = float(len(a_tokens))

        # 推理时拿不到 gold answer，因此这些“训练时可算、推理时不可直接算”的特征
        # 统一退化为 0，避免数据泄漏
        support = 0.0
        gold_in_passage = 0.0

        pred_in_passage = float(self.answer_in_passage(pred_answer, passage))
        q_p_overlap = float(self.overlap_ratio(question, passage))
        a_p_overlap = float(self.overlap_ratio(pred_answer, passage))

        return [
            float(bm25_score),         # bm25_score
            float(passage_rank),       # passage_rank
            question_len,              # question_len
            passage_len,               # passage_len
            pred_answer_len,           # pred_answer_len
            support,                   # support
            pred_in_passage,           # pred_answer_in_passage
            gold_in_passage,           # gold_answer_in_passage
            q_p_overlap,               # question_passage_overlap
            a_p_overlap,               # pred_answer_passage_overlap
        ]

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
                [self.extract_structured_features(
                    question=question,
                    passage=passage,
                    pred_answer=pred_answer,
                    bm25_score=bm25_score,
                    passage_rank=passage_rank,
                )],
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

        calibrated = prob ** 0.5
        return float(calibrated)

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

        probs = []
        for passage, pred_answer, bm25_score, rank in zip(
            passages, pred_answers, bm25_scores, passage_ranks
        ):
            prob = self.predict_one(
                question=question,
                passage=passage,
                pred_answer=pred_answer,
                bm25_score=bm25_score,
                passage_rank=rank,
            )
            probs.append(prob)

        return probs