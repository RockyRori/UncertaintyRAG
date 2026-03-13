#!/usr/bin/env bash

echo "=============================="
echo "Upgrading repo to Phase 3"
echo "Decision-Aware Retrieval"
echo "=============================="

dirs=(
  "controller"
  "decision"
  "retrieval"
  "uncertainty"
  "generator"
  "evaluation"
  "outputs"
)

files=(
  "controller/__init__.py"
  "controller/state.py"
  "controller/policy.py"

  "decision/__init__.py"
  "decision/actions.py"
  "decision/loop.py"

  "retrieval/__init__.py"
  "retrieval/rerank.py"

  "uncertainty/__init__.py"
  "uncertainty/signals.py"

  "generator/__init__.py"
  "generator/simple_answerer.py"

  "evaluation/__init__.py"
  "evaluation/decision_metrics.py"

  "main_phase3.py"
)

echo ""
echo "Creating directories..."
for dir in "${dirs[@]}"; do
  if [ ! -d "$dir" ]; then
    mkdir -p "$dir"
    echo "Created dir: $dir"
  else
    echo "Exists dir:  $dir"
  fi
done

echo ""
echo "Creating files..."
for file in "${files[@]}"; do
  if [ ! -f "$file" ]; then
    touch "$file"
    echo "Created file: $file"
  else
    echo "Exists file:  $file"
  fi
done

echo ""
echo "Phase 3 scaffold complete."