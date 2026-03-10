import random
from config import CORPUS_PATH, MINI_DATASET_PATH, RANDOM_SEED
from utils.io_utils import load_json, save_json
from utils.text_utils import contains_any_answer


QUESTIONS = [
    {"question": "What is the capital of France?", "gold_answers": ["Paris"]},
    {"question": "Which planet is the largest in the Solar System?", "gold_answers": ["Jupiter"]},
    {"question": "Who wrote Hamlet?", "gold_answers": ["William Shakespeare", "Shakespeare"]},
    {"question": "What is the chemical symbol for water?", "gold_answers": ["H2O"]},
    {"question": "What is the tallest mountain in the world?", "gold_answers": ["Mount Everest", "Everest"]},
    {"question": "Who painted the Mona Lisa?", "gold_answers": ["Leonardo da Vinci", "Da Vinci"]},
    {"question": "What is the capital of Japan?", "gold_answers": ["Tokyo"]},
    {"question": "Which animal is known as the king of the jungle?", "gold_answers": ["Lion", "The lion"]},
    {"question": "What gas do plants absorb from the atmosphere?", "gold_answers": ["Carbon dioxide", "CO2"]},
    {"question": "Who developed the theory of relativity?", "gold_answers": ["Albert Einstein", "Einstein"]},
]

NUM_SAMPLES = 100
PASSAGES_PER_QUESTION = 5


def extract_passage_text(p):
    if isinstance(p, str):
        return p
    if isinstance(p, dict):
        for key in ["text", "passage", "content", "body"]:
            if key in p:
                return str(p[key])
        raise KeyError(f"Cannot extract text from passage dict. Keys: {list(p.keys())}")
    return str(p)


def sample_passages_for_question(corpus_texts, gold_answers, k=5):
    positive_pool = [p for p in corpus_texts if contains_any_answer(p, gold_answers)]
    negative_pool = [p for p in corpus_texts if not contains_any_answer(p, gold_answers)]

    selected = []

    # 尽量保证至少1条正样本
    if positive_pool:
        selected.append(random.choice(positive_pool))

    remaining = k - len(selected)

    # 优先补负样本，模拟真实检索里“有对有错”
    if len(negative_pool) >= remaining:
        selected.extend(random.sample(negative_pool, remaining))
    else:
        selected.extend(negative_pool)
        remaining = k - len(selected)

        # 如果负样本不够，再补正样本
        extra_positive_pool = [p for p in positive_pool if p not in selected]
        if len(extra_positive_pool) >= remaining:
            selected.extend(random.sample(extra_positive_pool, remaining))
        else:
            selected.extend(extra_positive_pool)

    random.shuffle(selected)
    return selected[:k]


def main() -> None:
    random.seed(RANDOM_SEED)

    corpus = load_json(CORPUS_PATH)
    corpus_texts = [extract_passage_text(p) for p in corpus]

    mini_dataset = []

    for i in range(NUM_SAMPLES):
        qa = QUESTIONS[i % len(QUESTIONS)]
        question = qa["question"]
        gold_answers = qa["gold_answers"]

        passages = sample_passages_for_question(
            corpus_texts=corpus_texts,
            gold_answers=gold_answers,
            k=PASSAGES_PER_QUESTION
        )

        mini_dataset.append({
            "id": f"q_{i}",
            "question": question,
            "gold_answers": gold_answers,
            "passages": passages
        })

    save_json(mini_dataset, MINI_DATASET_PATH)
    print(f"Saved {len(mini_dataset)} samples to {MINI_DATASET_PATH}")


if __name__ == "__main__":
    main()