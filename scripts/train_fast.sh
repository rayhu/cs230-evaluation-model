#!/bin/bash
# Fast Training Configuration
# Quick iteration with improved performance over baseline

set -e

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Training FAST configuration..."

python scripts/train_mlp_regressor.py \
  --model-name sentence-transformers/all-MiniLM-L6-v2 \
  --epochs 50 \
  --batch-size 128 \
  --learning-rate 3e-4 \
  --hidden-dim1 384 \
  --hidden-dim2 192 \
  --dropout 0.25 \
  --val-split 0.15 \
  --output-dir experiments/mlp_fast \
  --device mps \
  --seed 42

echo ""
echo "Fast training complete! Expected MAE: ~0.065"

