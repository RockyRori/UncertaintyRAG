def exact_match(pred: str, gold: str) -> int:
    return int(pred.strip().lower() == gold.strip().lower())