import argparse
import random
from pathlib import Path

from config import DATASET_QA_PATH, RANDOM_SEED
from utils.io_utils import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--split", type=str, default="train", choices=["train", "dev", "all"])
    parser.add_argument("--output", type=str, default="data/mini_dataset.json")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    dataset = load_json(DATASET_QA_PATH)
    if not dataset:
        raise ValueError(f"No processed QA data found in {DATASET_QA_PATH}")

    if args.split != "all":
        dataset = [x for x in dataset if x.get("split") == args.split]

    if not dataset:
        raise ValueError(f"No QA records found for split={args.split}")

    if len(dataset) <= args.sample_size:
        sampled = dataset
    else:
        sampled = random.sample(dataset, args.sample_size)

    output_path = Path(args.output)
    save_json(sampled, output_path)
    print(f"Saved {len(sampled)} samples to {output_path}")


if __name__ == "__main__":
    main()