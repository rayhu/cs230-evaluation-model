#!/bin/bash
# Train MLP model using OpenAI embeddings
#
# Prerequisites:
# 1. Install OpenAI package: pip install openai python-dotenv
# 2. Set OPENAI_API_KEY environment variable or create .env file
#
# OpenAI Embedding Models:
# - text-embedding-3-small: 1536 dimensions, faster, cheaper (~$0.02/1M tokens)
# - text-embedding-3-large: 3072 dimensions, better quality (~$0.13/1M tokens)
# - text-embedding-ada-002: 1536 dimensions, legacy model

set -e

# Configuration
OPENAI_MODEL="${OPENAI_MODEL:-text-embedding-3-small}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/openai_embeddings4}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-0.0005}"
DROPOUT="${DROPOUT:-0.2}"
HIDDEN_DIM1="${HIDDEN_DIM1:-512}"
HIDDEN_DIM2="${HIDDEN_DIM2:-256}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
DEVICE="${DEVICE:-mps}"

echo "=============================================="
echo "Training with OpenAI Embeddings"
echo "=============================================="
echo "OpenAI Model: ${OPENAI_MODEL}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Epochs: ${EPOCHS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Dropout: ${DROPOUT}"
echo "Hidden Dims: ${HIDDEN_DIM1} -> ${HIDDEN_DIM2}"
echo "Weight Decay: ${WEIGHT_DECAY}"
echo "Device: ${DEVICE}"
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

# Run training
python scripts/train_mlp_regressor2.py \
    --use-openai-embeddings \
    --openai-model "${OPENAI_MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --dropout "${DROPOUT}" \
    --hidden-dim1 "${HIDDEN_DIM1}" \
    --hidden-dim2 "${HIDDEN_DIM2}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --device "${DEVICE}" \
    --use-improved-model \
    --use-hybrid-features

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "Results saved to: ${OUTPUT_DIR}"

