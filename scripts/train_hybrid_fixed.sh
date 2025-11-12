#!/bin/bash
# Train hybrid model with FIXED initialization + STRONG regularization
# 
# Key fixes:
#   1. Proper Kaiming initialization (prevents sigmoid saturation)
#   2. No dropout (hybrid features are sparse, dropout hurts)
#   3. STRONG L2 regularization (weight_decay=1e-3) to combat overfitting
#   4. ImprovedMLPRegressor with residual connections
#
# Overfitting issue: Train loss 0.0004 vs Val loss 0.006 (15x gap!)
# Solution: Increase weight_decay from 1e-4 to 1e-3 for stronger L2 penalty

python scripts/train_mlp_regressor.py \
    --model-name "sentence-transformers/all-mpnet-base-v2" \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0002 \
    --hidden-dim1 768 \
    --hidden-dim2 384 \
    --dropout 0.0 \
    --weight-decay 1e-3 \
    --val-split 0.15 \
    --output-dir "experiments/mlp_hybrid_regularized" \
    --device mps \
    --seed 42 \
    --use-hybrid-features \
    --use-improved-model

