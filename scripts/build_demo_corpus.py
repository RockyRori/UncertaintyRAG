import json
from pathlib import Path


def build_corpus():
    corpus = [
        {
            "id": "p1",
            "text": "Paris is the capital and most populous city of France."
        },
        {
            "id": "p2",
            "text": "William Shakespeare wrote many famous plays, including Hamlet, Macbeth, and Othello."
        },
        {
            "id": "p3",
            "text": "Jupiter is the largest planet in the Solar System."
        },
        {
            "id": "p4",
            "text": "Water boils at 100 degrees Celsius under standard atmospheric pressure."
        },
        {
            "id": "p5",
            "text": "Albert Einstein developed the theory of relativity."
        },
        {
            "id": "p6",
            "text": "Plants absorb carbon dioxide from the atmosphere during photosynthesis."
        },
        {
            "id": "p7",
            "text": "Mount Everest is the tallest mountain above sea level on Earth."
        },
        {
            "id": "p8",
            "text": "Leonardo da Vinci painted the Mona Lisa."
        },
        {
            "id": "p9",
            "text": "The chemical symbol for gold is Au."
        },
        {
            "id": "p10",
            "text": "The Pacific Ocean is the largest and deepest ocean on Earth."
        },

        # distractors
        {
            "id": "p11",
            "text": "Berlin is the capital of Germany."
        },
        {
            "id": "p12",
            "text": "Mars is often called the red planet."
        },
        {
            "id": "p13",
            "text": "Vincent van Gogh painted The Starry Night."
        },
        {
            "id": "p14",
            "text": "Silver has the chemical symbol Ag."
        },
        {
            "id": "p15",
            "text": "The Atlantic Ocean lies between the Americas and Europe and Africa."
        },
        {
            "id": "p16",
            "text": "Isaac Newton formulated the laws of motion and universal gravitation."
        },
        {
            "id": "p17",
            "text": "K2 is the second-highest mountain in the world."
        },
        {
            "id": "p18",
            "text": "Oxygen is essential for respiration in many living organisms."
        },
        {
            "id": "p19",
            "text": "Rome is the capital city of Italy."
        },
        {
            "id": "p20",
            "text": "Mercury is the smallest planet in the Solar System."
        }
    ]
    return corpus


def main():
    corpus = build_corpus()
    out_path = Path("data/corpus.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(corpus)} passages to {out_path}")


if __name__ == "__main__":
    main()