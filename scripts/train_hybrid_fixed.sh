#!/bin/bash
# Train hybrid model with FIXED hyperparameters
# Key fixes:
#   - Proper dropout (0.3 instead of 0.1)
#   - More epochs (100 with early stopping)
#   - Better architecture

python scripts/train_mlp_regressor.py \
    --model-name "sentence-transformers/all-mpnet-base-v2" \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0002 \
    --hidden-dim1 768 \
    --hidden-dim2 384 \
    --dropout 0.3 \
    --val-split 0.15 \
    --output-dir "experiments/mlp_hybrid_fixed" \
    --device mps \
    --seed 42 \
    --use-hybrid-features \
    --use-improved-model

