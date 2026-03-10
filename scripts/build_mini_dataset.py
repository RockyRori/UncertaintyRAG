import json
from pathlib import Path


def build_samples():
    base_samples = [
        {"question": "What is the capital of France?", "gold_answers": ["Paris"]},
        {"question": "Who wrote Hamlet?", "gold_answers": ["William Shakespeare", "Shakespeare"]},
        {"question": "Which planet is the largest in the Solar System?", "gold_answers": ["Jupiter"]},
        {"question": "What is the boiling point of water in Celsius?", "gold_answers": ["100", "100 degrees Celsius"]},
        {"question": "Who developed the theory of relativity?", "gold_answers": ["Albert Einstein", "Einstein"]},
        {"question": "What gas do plants absorb from the atmosphere?", "gold_answers": ["Carbon dioxide", "CO2"]},
        {"question": "What is the tallest mountain in the world?", "gold_answers": ["Mount Everest", "Everest"]},
        {"question": "Who painted the Mona Lisa?", "gold_answers": ["Leonardo da Vinci", "Da Vinci"]},
        {"question": "What is the chemical symbol for gold?", "gold_answers": ["Au"]},
        {"question": "What is the largest ocean on Earth?", "gold_answers": ["Pacific Ocean", "Pacific"]},
    ]

    samples = []
    for i in range(10):  # 10 * 10 = 100
        for j, item in enumerate(base_samples):
            samples.append({
                "id": f"q_{i}_{j}",
                "question": item["question"],
                "gold_answers": item["gold_answers"]
            })
    return samples


def main():
    data = build_samples()
    out_path = Path("data/mini_dataset.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} samples to {out_path}")


if __name__ == "__main__":
    main()