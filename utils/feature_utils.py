import re
from typing import Any, Dict, Iterable, List


SAFE_STRUCTURED_FEATURE_NAMES = [
    "bm25_score",
    "passage_rank",
    "question_len",
    "passage_len",
    "pred_answer_len",
    "pred_answer_in_passage",
    "question_passage_overlap",
    "pred_answer_passage_overlap",
]

LEGACY_STRUCTURED_FEATURE_NAMES = [
    "bm25_score",
    "passage_rank",
    "question_len",
    "passage_len",
    "pred_answer_len",
    "support",
    "pred_answer_in_passage",
    "gold_answer_in_passage",
    "question_passage_overlap",
    "pred_answer_passage_overlap",
]


def safe_text(x: Any) -> str:
    return "" if x is None else str(x)


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", safe_text(text).lower())


def overlap_ratio(a: str, b: str) -> float:
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    if not a_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(len(a_tokens), 1)


def answer_in_passage(pred_answer: str, passage: str) -> int:
    pred = safe_text(pred_answer).strip().lower()
    passage_low = safe_text(passage).lower()
    if not pred:
        return 0
    return int(pred in passage_low)


def build_text_feature_from_values(question: str, pred_answer: str, passage: str) -> str:
    return (
        f"question: {safe_text(question)} "
        f"[SEP] predicted_answer: {safe_text(pred_answer)} "
        f"[SEP] passage: {safe_text(passage)}"
    )


def runtime_feature_dict(
    question: str,
    passage: str,
    pred_answer: str,
    bm25_score: float = 0.0,
    passage_rank: int | float = 0,
) -> Dict[str, float]:
    question = safe_text(question)
    passage = safe_text(passage)
    pred_answer = safe_text(pred_answer)

    q_tokens = tokenize(question)
    p_tokens = tokenize(passage)
    a_tokens = tokenize(pred_answer)

    return {
        "bm25_score": float(bm25_score),
        "passage_rank": float(passage_rank),
        "question_len": float(len(q_tokens)),
        "passage_len": float(len(p_tokens)),
        "pred_answer_len": float(len(a_tokens)),
        "pred_answer_in_passage": float(answer_in_passage(pred_answer, passage)),
        "question_passage_overlap": float(overlap_ratio(question, passage)),
        "pred_answer_passage_overlap": float(overlap_ratio(pred_answer, passage)),
        # Legacy model compatibility only. These are unavailable at inference time
        # and must not be used by newly trained TPAMI models.
        "support": 0.0,
        "gold_answer_in_passage": 0.0,
    }


def structured_features_from_values(
    question: str,
    passage: str,
    pred_answer: str,
    bm25_score: float = 0.0,
    passage_rank: int | float = 0,
    feature_names: Iterable[str] | None = None,
) -> List[float]:
    names = list(feature_names or SAFE_STRUCTURED_FEATURE_NAMES)
    values = runtime_feature_dict(
        question=question,
        passage=passage,
        pred_answer=pred_answer,
        bm25_score=bm25_score,
        passage_rank=passage_rank,
    )
    return [float(values.get(name, 0.0)) for name in names]


def build_text_feature_from_sample(sample: Dict[str, Any]) -> str:
    return build_text_feature_from_values(
        question=sample.get("question", ""),
        pred_answer=sample.get("pred_answer", ""),
        passage=sample.get("passage", ""),
    )


def structured_features_from_sample(
    sample: Dict[str, Any],
    feature_names: Iterable[str] | None = None,
) -> List[float]:
    return structured_features_from_values(
        question=sample.get("question", ""),
        passage=sample.get("passage", ""),
        pred_answer=sample.get("pred_answer", ""),
        bm25_score=float(sample.get("bm25_score", sample.get("score", 0.0))),
        passage_rank=float(sample.get("passage_rank", sample.get("passage_index", 0))),
        feature_names=feature_names,
    )
