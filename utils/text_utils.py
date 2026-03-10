import re
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