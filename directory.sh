# =========================
# Phase 2A project scaffold
# =========================

# folders
mkdir data, retriever, generator, uncertainty, evaluation, utils, outputs, scripts -Force
mkdir data\raw, data\processed -Force

# package init
ni retriever\__init__.py -Force
ni generator\__init__.py -Force
ni uncertainty\__init__.py -Force
ni evaluation\__init__.py -Force
ni utils\__init__.py -Force

# root files
ni config.py -Force
ni main.py -Force

# retriever
ni retriever\bm25_retriever.py -Force
ni retriever\dense_retriever.py -Force

# generator
ni generator\qa_generator.py -Force

# uncertainty
ni uncertainty\scorer.py -Force

# evaluation
ni evaluation\metrics.py -Force

# utils
ni utils\io_utils.py -Force
ni utils\text_utils.py -Force

# scripts
ni scripts\build_demo_corpus.py -Force
ni scripts\build_mini_dataset.py -Force

# data files
ni data\mini_dataset.json -Force
ni data\corpus.json -Force

# output placeholders
ni outputs\predictions.jsonl -Force
ni outputs\metrics.json -Force
ni outputs\error_cases.json -Force