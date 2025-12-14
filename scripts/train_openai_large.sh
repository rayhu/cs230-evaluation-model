#!/bin/bash
# Train with OpenAI text-embedding-3-large (3072 dimensions)
# This uses the highest quality OpenAI embeddings with a deep model architecture.
#
# Note: text-embedding-3-large costs more than text-embedding-3-small
# Estimated cost: ~$0.13 per 1M tokens (vs ~$0.02 for small)

set -e

OUTPUT_DIR="${OUTPUT_DIR:-experiments/openai_large}"
EPOCHS="${EPOCHS:-200}"
DEVICE="${DEVICE:-mps}"

echo "=============================================="
echo "Training with OpenAI text-embedding-3-large"
echo "=============================================="
echo "This uses 3072-dimensional embeddings (highest quality)"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=============================================="

# Check for OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
    fi
    
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "ERROR: OPENAI_API_KEY not set!"
        exit 1
    fi
fi

# Run training with large model and deep architecture
python scripts/train_mlp_regressor.py \
    --use-openai-embeddings \
    --openai-model "text-embedding-3-large" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size 32 \
    --learning-rate 0.0003 \
    --dropout 0.25 \
    --weight-decay 5e-5 \
    --device "${DEVICE}" \
    --use-deep-model \
    --hidden-dims 1024 768 512 256 128 \
    --use-hybrid-features \
    --use-sample-weights

echo ""
echo "Training Complete! Results in: ${OUTPUT_DIR}"

