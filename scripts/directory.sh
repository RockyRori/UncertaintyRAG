#!/usr/bin/env bash
set -e

echo "===> Creating Phase 4 dataset-aware project structure..."

mkdir -p data/raw/squad
mkdir -p data/processed
mkdir -p data/demo
mkdir -p models/saved
mkdir -p outputs
mkdir -p scripts
mkdir -p datasets

touch datasets/__init__.py

echo "===> Done."
echo
echo "Expected structure:"
echo "data/raw/squad/           # raw SQuAD json files"
echo "data/processed/           # processed qa/corpus/utility json files"
echo "datasets/                 # future dataset loaders"
echo "scripts/                  # data prep / training scripts"
echo "models/saved/             # trained utility model + tfidf vectorizer"
echo "outputs/                  # predictions and metrics"