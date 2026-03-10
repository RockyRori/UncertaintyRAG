#!/usr/bin/env bash

echo "=============================="
echo "Setting up Phase 2B structure"
echo "=============================="

# directories
dirs=(
  "data"
  "models"
  "models/saved"
  "scripts"
  "utils"
  "training"
  "inference"
)

# files
files=(
  "models/__init__.py"
  "models/utility_predictor.py"

  "scripts/build_utility_dataset.py"

  "utils/__init__.py"
  "utils/io_utils.py"
  "utils/text_utils.py"

  "training/__init__.py"
  "training/train_utility_model.py"

  "inference/__init__.py"
  "inference/predict_utility.py"

  "config.py"
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
echo "Phase 2B scaffold complete."