# UncertaintyRAG TPAMI Prototype

This repository is a TPAMI-oriented prototype for uncertainty-guided adaptive retrieval-augmented question answering.

The current default is intentionally small: all main experiments use 100 examples so the pipeline can be debugged quickly before scaling up. All generated artifacts are written under `outputs/`, including downloaded data, processed JSON files, saved models, prediction files, result CSVs, paper tables, and figures.

## Current Defaults

- Dataset download limit per split: `100`
- Mini SQuAD training questions: `100`
- Utility training questions: `100`
- Evaluation samples: `100`
- Multi-dataset train/test split: `100/100`

Keep these defaults unchanged while debugging the pipeline. Change them in `config.py` only after data quality, split construction, and policy behavior are stable.

## Current Policy Notes

The default rule policy is calibrated for the current chunked-corpus outputs. It uses relaxed answer guardrails plus a utility-dominant answer rule: when the best candidate has strong predicted utility and total uncertainty is still acceptable, the controller may answer even if answer diversity across chunks raises conflict.

`sweep_policies.py` uses threshold bands derived from the current score distribution, from high-coverage settings to more selective settings. Rerun the sweep after utility-model, corpus, chunking, or answer-postprocessing changes.

## Clean Data Rule

Gold answers may be used to create supervision labels and evaluation metrics. They must not be copied into retrieval corpus text or runtime model features.

The utility model uses only inference-available structured features:

- `bm25_score`
- `passage_rank`
- `question_len`
- `passage_len`
- `pred_answer_len`
- `pred_answer_in_passage`
- `question_passage_overlap`
- `pred_answer_passage_overlap`

Open-domain datasets without clean evidence passages need an external retrieval corpus for serious experiments. Placeholder corpora are useful for debugging failure behavior, not for final benchmark claims.

## Run Directory

Run all commands from this folder:

```powershell
cd UncertaintyRAG
```

## First-Time Full Run

### 1. Environment

Install dependencies if the current environment has not been prepared yet.

```powershell
python -m pip install -r requirements.txt
```

### 2. SQuAD Single-Dataset Prototype

Use this path to build the main small SQuAD pipeline: processed data, mini dataset, utility supervision, utility model, governed decision loop, baseline comparison, policy sweep, and paper outputs.

```powershell
python -m scripts.prepare_squad --version v1.1
python -m scripts.build_mini_dataset --sample-size 100 --split train
python -m scripts.build_utility_dataset --max-questions 100 --top-k 3
python -m training.train_utility_model --max-questions 100
python -m scripts.diagnose_retrieval --datasets squad --split all --top-k 1 3 5
python main.py --max-samples 100
python compare_baselines.py --max-samples 100
python sweep_policies.py --max-samples 100
python -m scripts.build_results_csv
python -m scripts.export_paper_results
python -m scripts.plot_results
```

### 3. Multi-Dataset Prototype

Use this path to download and prepare the five small QA datasets, train per-dataset utility models, and run per-dataset evaluation.

```powershell
python -m scripts.download_qa_datasets --limit 100
python -m scripts.prepare_all_datasets
python -m scripts.diagnose_retrieval --datasets squad triviaqa webq --split test --top-k 1 3 5
python -m training.multi_dataset_runner --train-size 100 --test-size 100
python -m scripts.build_results_csv
python -m scripts.export_paper_results
python -m scripts.plot_results
```

The TriviaQA downloader tries multiple HuggingFace configurations and rejects empty streams. If one configuration returns no rows, it falls back automatically before writing final raw JSONL files.

By default, `multi_dataset_runner` skips datasets that cannot form non-overlapping train/test splits. Use strict mode when you want the run to fail instead of skipping invalid datasets:

```powershell
python -m training.multi_dataset_runner --train-size 100 --test-size 100 --strict-splits
```

## Command Categories

### Data Preparation

```powershell
python -m scripts.prepare_squad --version v1.1
python -m scripts.build_mini_dataset --sample-size 100 --split train
python -m scripts.download_qa_datasets --limit 100
python -m scripts.prepare_all_datasets
```

### Retrieval Diagnostics

```powershell
python -m scripts.diagnose_retrieval --datasets squad triviaqa webq --split test --top-k 1 3 5
```

### Utility Supervision and Training

```powershell
python -m scripts.build_utility_dataset --max-questions 100 --top-k 3
python -m training.train_utility_model --max-questions 100
python -m training.multi_dataset_runner --train-size 100 --test-size 100
```

### Evaluation

```powershell
python main.py --max-samples 100
python compare_baselines.py --max-samples 100
python sweep_policies.py --max-samples 100
```

### Result Aggregation and Paper Artifacts

```powershell
python -m scripts.build_results_csv
python -m scripts.export_paper_results
python -m scripts.plot_results
```

## Later Runs Without Starting Over

You do not need to rerun the full pipeline after every change. Use the smallest block that matches what changed.

### Only Policy or Decision Logic Changed

Run evaluation, policy sweep, and export again. No data preparation or utility retraining is needed.

```powershell
python main.py --max-samples 100
python compare_baselines.py --max-samples 100
python sweep_policies.py --max-samples 100
python -m scripts.build_results_csv
python -m scripts.export_paper_results
python -m scripts.plot_results
```

### Only Dataset Download, Corpus, or Chunking Changed

Regenerate processed data, check retrieval support, then rebuild utility supervision/models before evaluation. Use this block after changes to TriviaQA download, corpus construction, or chunking.

```powershell
python -m scripts.download_qa_datasets --limit 100
python -m scripts.prepare_all_datasets
python -m scripts.diagnose_retrieval --datasets squad triviaqa webq --split test --top-k 1 3 5
python -m scripts.prepare_squad --version v1.1
python -m scripts.build_mini_dataset --sample-size 100 --split train
python -m scripts.build_utility_dataset --max-questions 100 --top-k 3
python -m training.train_utility_model --max-questions 100
python main.py --max-samples 100
python compare_baselines.py --max-samples 100
python sweep_policies.py --max-samples 100
python -m training.multi_dataset_runner --train-size 100 --test-size 100
python -m scripts.build_results_csv
python -m scripts.export_paper_results
python -m scripts.plot_results
```

### Only Baseline Scripts Changed

```powershell
python compare_baselines.py --max-samples 100
python -m scripts.build_results_csv
python -m scripts.export_paper_results
python -m scripts.plot_results
```

### Only Policy Sweep Settings Changed

```powershell
python sweep_policies.py --max-samples 100
python -m scripts.build_results_csv
python -m scripts.export_paper_results
python -m scripts.plot_results
```

### Only Utility Model or Utility Features Changed

Rebuild utility supervision, retrain the utility model, then rerun evaluation and export.

```powershell
python -m scripts.build_utility_dataset --max-questions 100 --top-k 3
python -m training.train_utility_model --max-questions 100
python main.py --max-samples 100
python compare_baselines.py --max-samples 100
python sweep_policies.py --max-samples 100
python -m scripts.build_results_csv
python -m scripts.export_paper_results
python -m scripts.plot_results
```

### Only Tables or Figures Changed

No model or prediction rerun is needed.

```powershell
python -m scripts.build_results_csv
python -m scripts.export_paper_results
python -m scripts.plot_results
```

### Only Multi-Dataset Runner Changed

```powershell
python -m training.multi_dataset_runner --train-size 100 --test-size 100
python -m scripts.build_results_csv
python -m scripts.export_paper_results
python -m scripts.plot_results
```

## TPAMI Refactor Notes

The immediate goal is a reliable small-data pipeline, not final numbers. Next steps are:

- replace heuristic policy with learned cost-sensitive stopping/retrieval control;
- add a real answer-refinement reasoning action instead of treating rerank as reasoning;
- add dense/hybrid retrieval and reranker baselines;
- evaluate with matched budget and matched answer coverage;
- report calibration, risk-coverage, and error-type analysis.

## Average Rerun
```
python main.py --max-samples 100
python compare_baselines.py --max-samples 100
python sweep_policies.py --max-samples 100
python -m training.multi_dataset_runner --train-size 100 --test-size 100

python -m scripts.build_results_csv
python -m scripts.export_paper_results
python -m scripts.plot_results
```
