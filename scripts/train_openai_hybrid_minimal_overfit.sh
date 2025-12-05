#!/bin/bash
# Train hybrid model with MINIMAL overfitting using OpenAI embeddings
# 
# Prerequisites:
# 1. Install OpenAI package: pip install openai python-dotenv
# 2. Set OPENAI_API_KEY environment variable or create .env file
#
# OpenAI Embedding Model:
# - text-embedding-3-small: 1536 dimensions (~$0.02/1M tokens)
#
# Multi-pronged attack on overfitting:
#   1. SMALLER architecture: 512→256 (adjusted for 1536-dim OpenAI embeddings)
#   2. MODERATE dropout: 0.15 (small enough for sparse features)
#   3. STRONG L2: weight_decay=2e-3 (2x stronger than before)
#   4. Proper Kaiming initialization (from ImprovedMLPRegressor)
#
# Expected train/val gap: 2-3x (similar to sentence transformer version)
# Expected accuracy: maintained or slightly improved due to better embeddings

set -e

# Configuration - can be overridden via environment variables
OPENAI_MODEL="${OPENAI_MODEL:-text-embedding-3-small}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/openai_hybrid_minimal_overfit-final}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
DROPOUT="${DROPOUT:-0.15}"
HIDDEN_DIM1="${HIDDEN_DIM1:-512}"
HIDDEN_DIM2="${HIDDEN_DIM2:-256}"
WEIGHT_DECAY="${WEIGHT_DECAY:-2e-3}"
VAL_SPLIT="${VAL_SPLIT:-0.15}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"

echo "=============================================="
echo "Training with OpenAI Embeddings (Minimal Overfit)"
echo "=============================================="
echo "OpenAI Model:    ${OPENAI_MODEL}"
echo "Output Dir:      ${OUTPUT_DIR}"
echo "Epochs:          ${EPOCHS}"
echo "Batch Size:      ${BATCH_SIZE}"
echo "Learning Rate:   ${LEARNING_RATE}"
echo "Dropout:         ${DROPOUT}"
echo "Hidden Dims:     ${HIDDEN_DIM1} -> ${HIDDEN_DIM2}"
echo "Weight Decay:    ${WEIGHT_DECAY}"
echo "Val Split:       ${VAL_SPLIT}"
echo "Device:          ${DEVICE}"
echo "Seed:            ${SEED}"
echo "=============================================="

# Check for OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f ".env" ]; then
        echo "Loading API key from .env file..."
        export $(grep -v '^#' .env | xargs)
    fi
    
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "ERROR: OPENAI_API_KEY not set!"
        echo "Please set it via: export OPENAI_API_KEY=your-api-key"
        echo "Or create a .env file with: OPENAI_API_KEY=your-api-key"
        exit 1
    fi
fi

echo ""
echo "Starting training..."
echo ""

# Run training using train_mlp_regressor2.py (which supports OpenAI embeddings)
python scripts/train_mlp_regressor2.py \
    --use-openai-embeddings \
    --openai-model "${OPENAI_MODEL}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --hidden-dim1 "${HIDDEN_DIM1}" \
    --hidden-dim2 "${HIDDEN_DIM2}" \
    --dropout "${DROPOUT}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --val-split "${VAL_SPLIT}" \
    --output-dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --use-hybrid-features \
    --use-improved-model

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "To evaluate on test set, run:"
echo "  python scripts/evaluate_openai_model.py --model-dir ${OUTPUT_DIR}"

