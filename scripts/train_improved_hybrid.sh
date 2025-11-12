#!/bin/bash
# Train model with improved features and deep architecture (more layers, higher initial loss)

python scripts/train_mlp_regressor.py \
    --model-name "sentence-transformers/all-mpnet-base-v2" \
    --epochs 30 \
    --batch-size 64 \
    --learning-rate 0.0002 \
    --hidden-dim1 768 \
    --hidden-dim2 384 \
    --dropout 0.1 \
    --val-split 0.15 \
    --output-dir "experiments/mlp_hybrid_improved-0.1dropout" \
    --device mps \
    --seed 42 \
    --use-hybrid-features \
    --use-improved-model
    # --use-deep-model \
    # --hidden-dims 1536 1024 768 512 256 128

