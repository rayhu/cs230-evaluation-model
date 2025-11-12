#!/bin/bash
# Train hybrid model with SIMPLER architecture to reduce overfitting
# 
# Strategy: Reduce model capacity to match data complexity
#   - Smaller hidden layers (384 vs 768, 128 vs 384)
#   - Fewer parameters → less overfitting
#   - Moderate L2 regularization
#
# Expected: Better train/val loss ratio, similar or better validation performance

python scripts/train_mlp_regressor.py \
    --model-name "sentence-transformers/all-mpnet-base-v2" \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0003 \
    --hidden-dim1 384 \
    --hidden-dim2 128 \
    --dropout 0.0 \
    --weight-decay 5e-4 \
    --val-split 0.15 \
    --output-dir "experiments/mlp_hybrid_simple" \
    --device mps \
    --seed 42 \
    --use-hybrid-features \
    --use-improved-model

