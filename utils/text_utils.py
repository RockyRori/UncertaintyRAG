import re
from collections import Counter
from typing import List


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def contains_any_answer(passage: str, gold_answers: List[str]) -> bool:
    norm_passage = normalize_text(passage)
    for ans in gold_answers:
        if normalize_text(ans) in norm_passage:
            return True
    return False


def qa_match(pred: str, gold_answers: List[str]) -> int:
    pred_norm = normalize_text(pred)
    return int(any(pred_norm == normalize_text(g) for g in gold_answers))


def majority_answer(answers: List[str]) -> tuple[str, int]:
    cleaned = [normalize_text(a) for a in answers if a and normalize_text(a)]
    if not cleaned:
        return "", 0
    counter = Counter(cleaned)
    ans, count = counter.most_common(1)[0]
    return ans, count