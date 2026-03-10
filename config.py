from dataclasses import dataclass


@dataclass
class Config:
    # data
    dataset_path: str = "data/mini_dataset.json"
    corpus_path: str = "data/corpus.json"

    # retrieval
    top_k: int = 5

    # model
    model_name: str = "google/flan-t5-base"
    max_input_length: int = 512
    max_new_tokens: int = 32
    device: str = "cuda"

    # output
    predictions_path: str = "outputs/predictions.jsonl"
    metrics_path: str = "outputs/metrics.json"
    error_cases_path: str = "outputs/error_cases.json"

    # run
    use_demo_data: bool = True