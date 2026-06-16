import re
import string
from collections import Counter
from typing import Dict, List, Tuple


_ARTICLES = {"a", "an", "the"}
_WHO_VERBS = {
    "is",
    "was",
    "were",
    "set",
    "sets",
    "reached",
    "directed",
    "hosted",
    "won",
    "wrote",
    "founded",
    "became",
    "born",
    "died",
    "played",
    "invented",
    "discovered",
    "created",
    "starred",
    "scored",
    "served",
    "led",
}


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = [tok for tok in text.split() if tok not in _ARTICLES]
    return " ".join(tokens)


def _simple_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(text).lower())


def postprocess_answer(answer: str, question: str = "") -> str:
    """Clean generated snippets into short answer-like strings without gold labels."""
    if answer is None:
        return ""

    text = str(answer).strip()
    if not text:
        return ""

    text = re.sub(r"^\s*(answer|final answer)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n\"'")
    if not text:
        return ""

    question_norm = str(question or "").strip().lower()
    text_lower = text.lower()
    words = _simple_words(text_lower)

    if len(words) <= 8:
        return text.strip(" .;:,")

    if question_norm.startswith("who"):
        for idx, word in enumerate(words[1:8], start=1):
            if word in _WHO_VERBS and 1 <= idx <= 5:
                return " ".join(words[:idx]).strip()
            if word == "and" and idx >= 2:
                return " ".join(words[:idx]).strip()

    if question_norm.startswith(("which", "what")):
        m = re.match(r"^(.{2,80}?)\s+(is|was|were|are)\b", text_lower)
        if m:
            prefix_words = _simple_words(m.group(1))
            if 1 <= len(prefix_words) <= 6:
                return " ".join(prefix_words).strip()

    if "known for" in question_norm or "best known" in question_norm:
        m = re.search(r"\bfor\s+(?:his|her|its|their)?\s*([^.;,]+)", text_lower)
        if m:
            phrase = m.group(1)
            phrase = re.split(r"\b(published|in|by|at|on|with|who|which)\b", phrase)[0]
            phrase_words = _simple_words(phrase)
            if phrase_words:
                return " ".join(phrase_words[:6]).strip()

    first_clause = re.split(r"\s*(?:\.\.\.|[.;]|\s-\s)\s*", text)[0]
    first_clause = re.split(r"\s*,\s*(?:which|who|where|when|and)\b", first_clause, maxsplit=1)[0]
    clause_words = first_clause.split()
    if len(clause_words) > 12:
        first_clause = " ".join(clause_words[:12])
    return first_clause.strip(" .;:,")


def contains_any_answer(passage: str, gold_answers: List[str]) -> bool:
    norm_passage = normalize_text(passage)
    for ans in gold_answers:
        ans_norm = normalize_text(ans)
        if ans_norm and ans_norm in norm_passage:
            return True
    return False


def exact_match_score(pred: str, gold_answers: List[str]) -> int:
    pred_norm = normalize_text(pred)
    for gold in gold_answers:
        if pred_norm and pred_norm == normalize_text(gold):
            return 1
    return 0


def contains_answer_score(pred: str, gold_answers: List[str]) -> int:
    pred_norm = normalize_text(pred)
    if not pred_norm:
        return 0
    for gold in gold_answers:
        gold_norm = normalize_text(gold)
        if gold_norm and gold_norm in pred_norm:
            return 1
    return 0


def token_f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_token_f1(pred: str, gold_answers: List[str]) -> float:
    if not gold_answers:
        return 0.0
    return max(token_f1_score(pred, gold) for gold in gold_answers)


def qa_metrics(pred: str, gold_answers: List[str]) -> Dict[str, float]:
    return {
        "exact_match": float(exact_match_score(pred, gold_answers)),
        "contains_answer": float(contains_answer_score(pred, gold_answers)),
        "token_f1": float(max_token_f1(pred, gold_answers)),
    }


def qa_match(pred: str, gold_answers: List[str]) -> int:
    return exact_match_score(pred, gold_answers)


def majority_answer(answers: List[str]) -> Tuple[str, int]:
    cleaned = [normalize_text(a) for a in answers if normalize_text(a)]
    if not cleaned:
        return "", 0
    counter = Counter(cleaned)
    ans, count = counter.most_common(1)[0]
    return ans, count
