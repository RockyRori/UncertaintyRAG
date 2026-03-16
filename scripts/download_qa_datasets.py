from datasets import load_dataset
import json
from pathlib import Path
from itertools import islice

SAVE_DIR = Path("data/raw")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLE = 150


def save_jsonl(data_iter, path, n=N_SAMPLE):
    with open(path, "w", encoding="utf8") as f:
        for ex in islice(data_iter, n):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


print("Downloading SMALL streaming subsets...")

# -----------------
# SQuAD
# -----------------
ds = load_dataset("squad", split="train", streaming=True)
save_jsonl(ds, SAVE_DIR / "squad_train.jsonl")

ds = load_dataset("squad", split="validation", streaming=True)
save_jsonl(ds, SAVE_DIR / "squad_dev.jsonl")

# -----------------
# Natural Questions
# -----------------
ds = load_dataset("natural_questions", split="train", streaming=True)
save_jsonl(ds, SAVE_DIR / "nq.jsonl")

# -----------------
# TriviaQA
# -----------------
ds = load_dataset("trivia_qa", "rc", split="train", streaming=True)
save_jsonl(ds, SAVE_DIR / "trivia_train.jsonl")

ds = load_dataset("trivia_qa", "rc", split="validation", streaming=True)
save_jsonl(ds, SAVE_DIR / "trivia_dev.jsonl")

# -----------------
# WebQuestions
# -----------------
ds = load_dataset("web_questions", split="train", streaming=True)
save_jsonl(ds, SAVE_DIR / "webq_train.jsonl")

ds = load_dataset("web_questions", split="test", streaming=True)
save_jsonl(ds, SAVE_DIR / "webq_test.jsonl")

# -----------------
# PopQA
# -----------------
ds = load_dataset("akariasai/PopQA", split="test", streaming=True)
save_jsonl(ds, SAVE_DIR / "popqa_test.jsonl")


print("DONE. Only small subsets downloaded.")