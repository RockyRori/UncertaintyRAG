import re
import string


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.lower().strip()

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(pred: str, gold_answers: list[str]) -> int:
    pred_norm = normalize_text(pred)
    for ans in gold_answers:
        if pred_norm == normalize_text(ans):
            return 1
    return 0


def contains_match(pred: str, gold_answers: list[str]) -> int:
    pred_norm = normalize_text(pred)
    for ans in gold_answers:
        ans_norm = normalize_text(ans)
        if ans_norm and (ans_norm in pred_norm or pred_norm in ans_norm):
            return 1
    return 0


def qa_match(pred: str, gold_answers: list[str]) -> int:
    return 1 if (exact_match(pred, gold_answers) or contains_match(pred, gold_answers)) else 0