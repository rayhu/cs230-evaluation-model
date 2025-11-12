#!/bin/bash
# Train hybrid model with SIMPLER architecture to reduce overfitting
# 
# Strategy: Reduce model capacity to match data complexity
#   - Smaller hidden layers: 384→128 (vs 768→384)
#   - Parameters: ~400K (vs 1.2M) - 3x smaller!
#   - Strong L2 regularization (1e-3)
#   - No dropout (sparse features)
#
# Expected: 
#   - Train/val loss ratio: 3-5x (vs current 15x)
#   - Similar accuracy (55-60% at ±5%)
#   - Better generalization

python scripts/train_mlp_regressor.py \
    --model-name "sentence-transformers/all-mpnet-base-v2" \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0002 \
    --hidden-dim1 384 \
    --hidden-dim2 128 \
    --dropout 0.0 \
    --weight-decay 1e-3 \
    --val-split 0.15 \
    --output-dir "experiments/mlp_hybrid_simple" \
    --device mps \
    --seed 42 \
    --use-hybrid-features \
    --use-improved-model


