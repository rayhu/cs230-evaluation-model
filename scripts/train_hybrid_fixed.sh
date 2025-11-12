#!/bin/bash
# Train hybrid model with FIXED weight initialization
# 
# Key fixes:
#   1. Proper Kaiming initialization (prevents sigmoid saturation with high-dim inputs)
#   2. No dropout (hybrid features are already diverse)
#   3. ImprovedMLPRegressor with residual connections
#
# Previous issue: std=0.8 weight init caused saturation with 807-dim inputs
# Fix: Kaiming init scales by sqrt(2/fan_in) ≈ 0.05 for 807 inputs

python scripts/train_mlp_regressor.py \
    --model-name "sentence-transformers/all-mpnet-base-v2" \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0002 \
    --hidden-dim1 768 \
    --hidden-dim2 384 \
    --dropout 0.0 \
    --val-split 0.15 \
    --output-dir "experiments/mlp_hybrid_proper_init" \
    --device mps \
    --seed 42 \
    --use-hybrid-features \
    --use-improved-model

