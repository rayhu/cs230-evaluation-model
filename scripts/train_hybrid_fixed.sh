#!/bin/bash
# Train hybrid model with CORRECT hyperparameters
# Key insight: Hybrid features work BEST with NO DROPOUT!
# The features are already diverse (structure + text + embeddings)
# Adding dropout destroys information in sparse features

python scripts/train_mlp_regressor.py \
    --model-name "sentence-transformers/all-mpnet-base-v2" \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0002 \
    --hidden-dim1 768 \
    --hidden-dim2 384 \
    --dropout 0.0 \
    --val-split 0.15 \
    --output-dir "experiments/mlp_hybrid_best" \
    --device mps \
    --seed 42 \
    --use-hybrid-features \
    --use-improved-model

