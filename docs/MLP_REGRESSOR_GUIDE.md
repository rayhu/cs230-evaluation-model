# MLP Regressor for Table Quality Prediction

## Overview

This guide explains how to train and use the simple MLP (Multi-Layer Perceptron) regressor that predicts table extraction quality scores **without requiring ground truth**.

**Key Features:**
- **No Ground Truth Needed**: Predicts quality scores from extracted table JSON alone
- **TF-IDF + Neural Network**: Combines classical NLP with deep learning
- **Fast Training**: Trains in minutes on CPU
- **Easy Deployment**: Single file prediction with saved model

## Architecture

```
Input: Generated Table JSON (as text)
   ↓
TF-IDF Vectorization (max 10,000 features)
   ↓
MLP: Linear(10000, 256) → ReLU → Linear(256, 64) → ReLU → Linear(64, 1)
   ↓
Output: Predicted Quality Score (0-1)
```

## Quick Start

### 1. Train the Model

Train on the full dataset (11,971 training examples):

```bash
# Activate virtual environment
source .venv/bin/activate

# Train with default settings
python scripts/train_mlp_regressor.py \
  --epochs 10 \
  --output-dir experiments/mlp_regressor

# Or train with custom hyperparameters
python scripts/train_mlp_regressor.py \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --hidden-dim1 512 \
  --hidden-dim2 128 \
  --dropout 0.2 \
  --output-dir experiments/mlp_custom
```

**Training time:**
- CPU: ~5-10 minutes for 10 epochs on full dataset
- MPS (Apple Silicon): ~2-3 minutes
- CUDA: ~1-2 minutes

### 2. Predict Quality Scores

**Single file prediction:**
```bash
python scripts/predict_quality.py \
  --model-dir experiments/mlp_regressor \
  --input data/SciTSR/test/json_output/0704.1068v2.1.json
```

**Batch prediction:**
```bash
python scripts/predict_quality.py \
  --model-dir experiments/mlp_regressor \
  --input data/SciTSR/test/json_output \
  --output results/mlp_predictions.json
```

## Training Options

```bash
python scripts/train_mlp_regressor.py --help
```

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--limit` | None | Limit training samples (for testing) |
| `--max-features` | 10000 | Max TF-IDF features |
| `--epochs` | 10 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--learning-rate` | 0.001 | Learning rate |
| `--hidden-dim1` | 256 | First hidden layer size |
| `--hidden-dim2` | 64 | Second hidden layer size |
| `--dropout` | 0.0 | Dropout rate (0 = no dropout) |
| `--val-split` | 0.2 | Validation split ratio |
| `--output-dir` | experiments/mlp_regressor | Output directory |
| `--device` | cpu | Device: cpu, cuda, or mps |

## Output Files

After training, the output directory contains:

```
experiments/mlp_regressor/
├── best_model.pt              # Best model checkpoint (lowest val MAE)
├── final_model.pt             # Final model after all epochs
├── tfidf_vectorizer.pkl       # Fitted TF-IDF vectorizer
├── training_history.json      # Loss and metrics per epoch
└── config.json                # Training configuration
```

## Example: Complete Workflow

### Step 1: Quick Test (Small Dataset)

```bash
# Test on 100 samples, 3 epochs
python scripts/train_mlp_regressor.py \
  --limit 100 \
  --epochs 3 \
  --max-features 1000 \
  --output-dir experiments/mlp_test

# Expected output:
# Epoch 1/3 - Train Loss: 0.1968, Val MAE: 0.4168
# Epoch 2/3 - Train Loss: 0.1560, Val MAE: 0.3659
# Epoch 3/3 - Train Loss: 0.1211, Val MAE: 0.3062
```

### Step 2: Full Training

```bash
# Train on full dataset with optimized settings
python scripts/train_mlp_regressor.py \
  --epochs 20 \
  --batch-size 64 \
  --hidden-dim1 512 \
  --hidden-dim2 128 \
  --dropout 0.2 \
  --device mps \
  --output-dir experiments/mlp_full
```

### Step 3: Evaluate on Test Set

```bash
# Predict quality for all test files
python scripts/predict_quality.py \
  --model-dir experiments/mlp_full \
  --input data/SciTSR/test/json_output \
  --output results/mlp_test_predictions.json

# Output shows:
# - Per-file predictions
# - Average score
# - Min/Max scores
# - Standard deviation
```

### Step 4: Compare Predictions vs Ground Truth

```python
import json
import numpy as np

# Load predictions
with open('results/mlp_test_predictions.json') as f:
    preds = json.load(f)

# Load ground truth scores (from score_extraction.py)
with open('results/evaluation_scores.json') as f:
    gt_scores = json.load(f)

# Compare
pred_scores = {p['filename']: p['predicted_score'] for p in preds['predictions']}
gt_individual = {s['filename']: s['overall_score'] for s in gt_scores['individual_scores']}

# Calculate correlation
errors = []
for filename in pred_scores:
    if filename in gt_individual:
        pred = pred_scores[filename]
        gt = gt_individual[filename]
        errors.append(abs(pred - gt))

print(f"Mean Absolute Error: {np.mean(errors):.4f}")
print(f"RMSE: {np.sqrt(np.mean([e**2 for e in errors])):.4f}")
```

## Hyperparameter Tuning

### Recommended Configurations

**Fast Training (Baseline):**
```bash
--epochs 10 --batch-size 32 --hidden-dim1 256 --hidden-dim2 64
```

**Better Performance:**
```bash
--epochs 20 --batch-size 64 --hidden-dim1 512 --hidden-dim2 128 --dropout 0.2
```

**Regularization (Prevent Overfitting):**
```bash
--epochs 30 --dropout 0.3 --learning-rate 0.0005
```

### Grid Search Example

```bash
for hidden1 in 256 512 1024; do
  for hidden2 in 64 128 256; do
    for dropout in 0.0 0.1 0.2; do
      python scripts/train_mlp_regressor.py \
        --epochs 20 \
        --hidden-dim1 $hidden1 \
        --hidden-dim2 $hidden2 \
        --dropout $dropout \
        --output-dir experiments/grid_h1${hidden1}_h2${hidden2}_d${dropout}
    done
  done
done
```

## Performance Metrics

The model is evaluated using:

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and true scores
- **RMSE (Root Mean Squared Error)**: Square root of average squared errors
- **MSE (Mean Squared Error)**: Average squared differences

**Typical Results** (on validation set):
- MAE: 0.05-0.15 (depending on hyperparameters)
- RMSE: 0.08-0.20

## Use Cases

### 1. Real-time Quality Assessment

Predict quality scores for new extractions without waiting for manual evaluation:

```python
import json
from predict_quality import load_model_and_vectorizer, predict_single

model, vectorizer = load_model_and_vectorizer('experiments/mlp_regressor')

with open('new_extraction.json') as f:
    table = json.load(f)

score = predict_single(model, vectorizer, table)
print(f"Predicted Quality: {score:.2%}")
```

### 2. Batch Processing

Evaluate large batches of extractions:

```bash
# Process 3000 test files
python scripts/predict_quality.py \
  --model-dir experiments/mlp_regressor \
  --input data/SciTSR/test/json_output \
  --output results/batch_predictions.json
```

### 3. Active Learning

Use predictions to prioritize which extractions need manual review:

```python
predictions = json.load(open('results/batch_predictions.json'))

# Flag low-quality predictions for review
low_quality = [p for p in predictions['predictions'] 
               if p['predicted_score'] < 0.3]

print(f"Found {len(low_quality)} low-quality extractions")
```

## Troubleshooting

### Issue: Out of Memory

**Solution:** Reduce batch size or TF-IDF features:
```bash
--batch-size 16 --max-features 5000
```

### Issue: Overfitting (train loss << val loss)

**Solution:** Add dropout and reduce model size:
```bash
--dropout 0.3 --hidden-dim1 128 --hidden-dim2 32
```

### Issue: Underfitting (both losses high)

**Solution:** Increase model capacity:
```bash
--hidden-dim1 512 --hidden-dim2 256 --epochs 30
```

### Issue: Slow Training

**Solution:** Use GPU/MPS:
```bash
--device mps  # For Apple Silicon
--device cuda # For NVIDIA GPU
```

## Integration with Existing Pipeline

The MLP regressor integrates seamlessly with the existing evaluation pipeline:

```bash
# 1. Extract tables
python scripts/extract_tables_scitsr.py \
  --input data/SciTSR/test/img \
  --output data/SciTSR/test/json_output

# 2. Predict quality (NO GROUND TRUTH NEEDED)
python scripts/predict_quality.py \
  --model-dir experiments/mlp_regressor \
  --input data/SciTSR/test/json_output \
  --output results/predicted_scores.json

# 3. (Optional) Validate with ground truth
python scripts/score_extraction.py \
  --pred data/SciTSR/test/json_output \
  --gt data/SciTSR/test/structure \
  --output results/ground_truth_scores.json
```

## Next Steps

1. **Train on Full Dataset**: Run with `--epochs 20` on full data
2. **Hyperparameter Tuning**: Try different architectures
3. **Feature Engineering**: Experiment with different `max_features` values
4. **Ensemble**: Train multiple models and average predictions
5. **Advanced Models**: Try GRU/LSTM for sequence modeling of cell structures

## References

- **Dataset**: [rayhu/table-extraction-evaluation](https://huggingface.co/datasets/rayhu/table-extraction-evaluation)
- **TF-IDF**: Scikit-learn documentation
- **PyTorch**: [pytorch.org](https://pytorch.org)

