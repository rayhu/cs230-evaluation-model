#!/bin/bash
# Train hybrid model with BALANCED regularization
# 
# Strategy: Combine multiple regularization techniques
#   1. Small dropout (0.2) - careful amount that won't destroy sparse features
#   2. Strong L2 (weight_decay=1e-3) - penalize large weights
#   3. Kaiming initialization - prevents saturation
#   4. Early stopping - prevents overtraining
#
# Previous attempts:
#   - dropout=0.0, weight_decay=1e-4 → overfitting (train/val gap 15x)
#   - dropout=0.0, weight_decay=1e-3 → still overfitting (train/val gap 15x)
#   - dropout=0.1 with bad init → complete failure (R²=-20)
#
# This time: dropout=0.2 + weight_decay=1e-3 + proper init

python scripts/train_mlp_regressor.py \
    --model-name "sentence-transformers/all-mpnet-base-v2" \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0002 \
    --hidden-dim1 768 \
    --hidden-dim2 384 \
    --dropout 0.2 \
    --weight-decay 1e-3 \
    --val-split 0.15 \
    --output-dir "experiments/mlp_hybrid_balanced" \
    --device mps \
    --seed 42 \
    --use-hybrid-features \
    --use-improved-model

