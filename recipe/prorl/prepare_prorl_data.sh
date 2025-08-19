#!/bin/bash
# ProRL Data Preparation Script
# Prepares diverse task datasets for ProRL training

set -e

# Configuration
DATA_DIR="${HOME}/verl/data/prorl_diverse_tasks"
SCRIPT_DIR="$(dirname "$0")"

echo "=== ProRL Data Preparation ==="
echo "Output directory: $DATA_DIR"

# Create data directory
mkdir -p "$DATA_DIR"

# Run data preparation
echo "Generating diverse task datasets..."

python "$SCRIPT_DIR/prepare_prorl_data.py" \
    --output_dir "$DATA_DIR" \
    --math_samples 10000 \
    --code_samples 5000 \
    --stem_samples 3000 \
    --logic_samples 2000 \
    --instruction_samples 3000 \
    --val_samples 1000

echo ""
echo "=== Data Preparation Complete ==="
echo "Datasets created in: $DATA_DIR"
echo ""
echo "Dataset summary:"
wc -l "$DATA_DIR"/*.jsonl

echo ""
echo "You can now run ProRL training with:"
echo "  bash run_prorl_qwen_1.5b.sh"