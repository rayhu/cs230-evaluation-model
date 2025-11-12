#!/bin/bash
# Improved MLP Regressor Training Script
# This configuration addresses the issues in the baseline model

set -e

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=========================================="
echo "Training Improved MLP Regressor"
echo "=========================================="
echo ""
echo "Improvements over baseline:"
echo "  ✓ Better sentence transformer (mpnet: 768-dim vs MiniLM: 384-dim)"
echo "  ✓ Proper dropout regularization (0.3)"
echo "  ✓ More epochs with early stopping (100 vs 10)"
echo "  ✓ Larger validation set (15% vs 10%)"
echo "  ✓ Optimized architecture for 768-dim embeddings"
echo "  ✓ Lower learning rate for stability"
echo ""

# Configuration
MODEL_NAME="sentence-transformers/all-mpnet-base-v2"  # Better model (768-dim)
EPOCHS=100                                              # More training with early stopping
BATCH_SIZE=64                                          # Good balance
LEARNING_RATE=2e-4                                     # Lower for stability
HIDDEN_DIM1=768                                        # Match embedding dimension
HIDDEN_DIM2=384                                        # Gradual reduction
DROPOUT=0.3                                            # Regularization
VAL_SPLIT=0.15                                         # Larger validation set
OUTPUT_DIR="experiments/mlp_sentence_transformer_improved"
DEVICE="mps"                                           # Use Metal Performance Shaders (Mac)

python scripts/train_mlp_regressor.py \
  --model-name "$MODEL_NAME" \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --learning-rate $LEARNING_RATE \
  --hidden-dim1 $HIDDEN_DIM1 \
  --hidden-dim2 $HIDDEN_DIM2 \
  --dropout $DROPOUT \
  --val-split $VAL_SPLIT \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --seed 42

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "To evaluate on test set, run:"
echo "  python scripts/evaluate_on_test_set.py \\"
echo "    --model-dir $OUTPUT_DIR \\"
echo "    --output $OUTPUT_DIR/test_evaluation.json \\"
echo "    --plot-dir $OUTPUT_DIR/plots"

