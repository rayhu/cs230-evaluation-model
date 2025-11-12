#!/bin/bash
# Maximum Accuracy Configuration
# Best possible performance with longer training time

set -e

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Training MAXIMUM ACCURACY configuration..."
echo "This will take longer but should achieve best results"

python scripts/train_mlp_regressor.py \
  --model-name sentence-transformers/all-mpnet-base-v2 \
  --epochs 150 \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --hidden-dim1 1024 \
  --hidden-dim2 512 \
  --dropout 0.35 \
  --val-split 0.2 \
  --output-dir experiments/mlp_maximum_accuracy \
  --device mps \
  --seed 42

echo ""
echo "Maximum accuracy training complete! Expected MAE: <0.055"

