#!/bin/bash
# Train hybrid model with MINIMAL overfitting
# 
# Multi-pronged attack on overfitting:
#   1. SMALLER architecture: 384→128 (vs 768→384) - 3x fewer params
#   2. MODERATE dropout: 0.15 (small enough for sparse features)
#   3. STRONG L2: weight_decay=2e-3 (2x stronger than before)
#   4. Proper Kaiming initialization
#
# Expected train/val gap: 2-3x (current: 3.5x, before: 18x)
# Expected accuracy: maintained or slightly improved

python scripts/train_mlp_regressor.py \
    --model-name "sentence-transformers/all-mpnet-base-v2" \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0002 \
    --hidden-dim1 384 \
    --hidden-dim2 128 \
    --dropout 0.15 \
    --weight-decay 2e-3 \
    --val-split 0.15 \
    --output-dir "experiments/updated_data_mlp_hybrid_minimal_overfit" \
    --device mps \
    --seed 42 \
    --use-hybrid-features \
    --use-improved-model

