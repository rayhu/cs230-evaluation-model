# MLP Regressor Quick Reference

## 🚀 Installation

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install scikit-learn
```

## 📝 Training

### Quick Test (100 samples, 3 epochs)
```bash
python scripts/train_mlp_regressor.py \
  --limit 100 \
  --epochs 3 \
  --max-features 1000 \
  --output-dir experiments/mlp_test
```

### Full Training (Recommended)
```bash
python scripts/train_mlp_regressor.py \
  --epochs 20 \
  --batch-size 64 \
  --hidden-dim1 512 \
  --hidden-dim2 128 \
  --dropout 0.2 \
  --output-dir experiments/mlp_full
```

### Fast CPU Training
```bash
python scripts/train_mlp_regressor.py \
  --epochs 10 \
  --output-dir experiments/mlp_regressor
```

### GPU/MPS Training
```bash
python scripts/train_mlp_regressor.py \
  --epochs 20 \
  --device mps \
  --output-dir experiments/mlp_gpu
```

## 🔮 Prediction

### Single File
```bash
python scripts/predict_quality.py \
  --model-dir experiments/mlp_regressor \
  --input data/SciTSR/test/json_output/0704.1068v2.1.json
```

### Batch Prediction
```bash
python scripts/predict_quality.py \
  --model-dir experiments/mlp_regressor \
  --input data/SciTSR/test/json_output \
  --output results/predictions.json
```

## 📊 Complete Workflow

```bash
# 1. Extract tables (if not already done)
python scripts/extract_tables_scitsr.py \
  --input data/SciTSR/test/img \
  --output data/SciTSR/test/json_output

# 2. Train model
python scripts/train_mlp_regressor.py \
  --epochs 10 \
  --output-dir experiments/mlp_regressor

# 3. Predict quality scores
python scripts/predict_quality.py \
  --model-dir experiments/mlp_regressor \
  --input data/SciTSR/test/json_output \
  --output results/mlp_predictions.json
```

## ⚙️ Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--limit` | None | Limit samples (for testing) |
| `--epochs` | 10 | Number of epochs |
| `--batch-size` | 32 | Batch size |
| `--learning-rate` | 0.001 | Learning rate |
| `--hidden-dim1` | 256 | First hidden layer |
| `--hidden-dim2` | 64 | Second hidden layer |
| `--dropout` | 0.0 | Dropout rate |
| `--max-features` | 10000 | TF-IDF features |
| `--device` | cpu | cpu/cuda/mps |

## 📁 Output Files

```
experiments/mlp_regressor/
├── best_model.pt              # Best model (lowest val MAE)
├── final_model.pt             # Final model
├── tfidf_vectorizer.pkl       # TF-IDF vectorizer
├── training_history.json      # Training metrics
└── config.json                # Configuration
```

## 🎯 Expected Performance

- **Validation MAE**: 0.05-0.15
- **Training Time**: 5-10 min (CPU), 1-2 min (GPU)
- **Model Size**: ~2.5 MB

## 📖 Full Documentation

- **Complete Guide**: [`docs/MLP_REGRESSOR_GUIDE.md`](docs/MLP_REGRESSOR_GUIDE.md)
- **Implementation Summary**: [`experiments/MLP_IMPLEMENTATION_SUMMARY.md`](experiments/MLP_IMPLEMENTATION_SUMMARY.md)
- **Dataset Usage**: [`DATASET_USAGE.md`](DATASET_USAGE.md)

