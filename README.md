# UncertaintyRAG TPAMI Prototype

This repository is being refactored from a conference prototype into a TPAMI-oriented experimental codebase for uncertainty-guided adaptive RAG.

The current default is intentionally small: training is capped at 100 questions so the full loop can be debugged quickly before scaling up.

## Current Defaults

- Dataset download limit per split: `100`
- Mini training questions: `100`
- Utility training questions: `100`
- Evaluation samples: `100`
- Multi-dataset train/test split: `100/100`

Change these in `config.py` only after the pipeline is stable.

## Clean Data Rule

Gold answers may be used to create supervision labels and evaluation metrics. They must not be copied into retrieval corpus text or runtime model features.

The utility model now uses only inference-available structured features:

- `bm25_score`
- `passage_rank`
- `question_len`
- `passage_len`
- `pred_answer_len`
- `pred_answer_in_passage`
- `question_passage_overlap`
- `pred_answer_passage_overlap`

Legacy saved models with older feature bundles are still readable, but newly trained models use the safe feature list.

## Quick SQuAD Prototype

Run commands from this directory.

```powershell
python -m scripts.prepare_squad --version v1.1
python -m scripts.build_mini_dataset --sample-size 100 --split train
python -m scripts.build_utility_dataset --max-questions 100 --top-k 3
python -m training.train_utility_model --max-questions 100
python main.py --max-samples 100
python main_compare_phase5.py --max-samples 100
python main_sweep_phase5.py --max-samples 100
```

## Small Multi-Dataset Prototype

The helper below downloads at most 100 examples per split. Open-domain datasets without clean evidence passages need an external retrieval corpus for serious experiments.

```powershell
python -m scripts.download_qa_datasets --limit 100
python -m scripts.prepare_all_datasets
python -m training.multi_dataset_runner --train-size 100 --test-size 100
```

## Result Export

After prediction files are generated:

```powershell
python -m scripts.build_results_csv_v2
python -m scripts.export_paper_results_v2
python -m scripts.plot_results_v2
```

## TPAMI Refactor Notes

The immediate goal is a reliable small-data pipeline, not final numbers. Next steps are:

- replace heuristic policy with learned cost-sensitive stopping/retrieval control;
- add a real answer-refinement reasoning action instead of treating rerank as reasoning;
- add dense/hybrid retrieval and reranker baselines;
- evaluate with matched budget and matched answer coverage;
- report calibration, risk-coverage, and error-type analysis.
