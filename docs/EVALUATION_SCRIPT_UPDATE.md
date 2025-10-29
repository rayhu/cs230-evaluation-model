# ✅ Evaluation Script Updated for Word2Vec

## What Was Fixed

The `scripts/evaluate_on_test_set.py` script now **automatically detects and supports both**:
- ✅ **Word2Vec models** (new)
- ✅ **TF-IDF models** (legacy)

## Changes Made

### 1. Auto-Detection of Model Type

The script now checks for both model types:

```python
word2vec_path = args.model_dir / 'word2vec_model.bin'
tfidf_path = args.model_dir / 'tfidf_vectorizer.pkl'

if word2vec_path.exists():
    # Load Word2Vec + StandardScaler
elif tfidf_path.exists():
    # Load TF-IDF (legacy)
else:
    # Error: no model found
```

### 2. Word2Vec Feature Extraction

Added functions to handle Word2Vec:

```python
def tokenize_json(text: str):
    """Tokenize JSON text into words."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def doc_to_vec(tokens, w2v_model):
    """Convert document tokens to average Word2Vec vector."""
    vectors = []
    for word in tokens:
        if word in w2v_model.wv:
            vectors.append(w2v_model.wv[word])
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)
```

### 3. Unified Prediction Function

Updated to handle both model types:

```python
if use_word2vec:
    # Word2Vec: tokenize → vector → scale
    tokens = tokenize_json(text)
    vec = doc_to_vec(tokens, feature_extractor)
    features = scaler.transform([vec])
else:
    # TF-IDF: direct transform
    features = feature_extractor.transform([text]).toarray()
```

### 4. Enhanced Output

Now includes model type in results:

```python
results = {
    'model_dir': str(args.model_dir),
    'model_type': 'word2vec' if use_word2vec else 'tfidf',  # ← NEW
    'test_set_size': len(ground_truth),
    'num_predictions': len(predictions),
    'metrics': metrics,
    'analysis': analysis
}
```

## How to Use

### For Word2Vec Models (New)

```bash
source .venv/bin/activate

# Evaluate Word2Vec model
python scripts/evaluate_on_test_set.py \
  --model-dir experiments/mlp_word2vec_full \
  --output results/word2vec_evaluation.json \
  --plot-dir results/word2vec_plots \
  --device mps
```

**Expected output:**
```
======================================================================
LOADING MODEL
======================================================================
Detected Word2Vec model
✓ Loaded Word2Vec model from experiments/mlp_word2vec_full/word2vec_model.bin
  Vocabulary size: 62991
  Vector dimension: 300
✓ Loaded feature scaler from experiments/mlp_word2vec_full/feature_scaler.pkl
✓ Loaded model from experiments/mlp_word2vec_full/best_model.pt
  Architecture: 300 -> 256 -> 64 -> 1

======================================================================
GENERATING PREDICTIONS
======================================================================
Generating predictions using Word2Vec...
Predicting: 100% 3000/3000
```

### For TF-IDF Models (Legacy)

```bash
# Evaluate old TF-IDF model
python scripts/evaluate_on_test_set.py \
  --model-dir experiments/mlp_deep \
  --output results/tfidf_evaluation.json \
  --plot-dir results/tfidf_plots \
  --device cpu
```

**Expected output:**
```
======================================================================
LOADING MODEL
======================================================================
Detected TF-IDF model (legacy)
✓ Loaded TF-IDF vectorizer from experiments/mlp_deep/tfidf_vectorizer.pkl
✓ Loaded model from experiments/mlp_deep/best_model.pt
  Architecture: 10000 -> 256 -> 64 -> 1

======================================================================
GENERATING PREDICTIONS
======================================================================
Generating predictions using TF-IDF...
Predicting: 100% 3000/3000
```

## Required Files

### Word2Vec Model Directory Must Contain:
- ✅ `word2vec_model.bin` - Trained Word2Vec embeddings
- ✅ `feature_scaler.pkl` - StandardScaler for normalization
- ✅ `best_model.pt` or `final_model.pt` - Trained MLP model

### TF-IDF Model Directory Must Contain:
- ✅ `tfidf_vectorizer.pkl` - Fitted TF-IDF vectorizer
- ✅ `best_model.pt` or `final_model.pt` - Trained MLP model

## Error Handling

If neither model type is found:

```
FileNotFoundError: No feature extractor found in experiments/some_dir
Expected either:
  - experiments/some_dir/word2vec_model.bin (Word2Vec)
  - experiments/some_dir/tfidf_vectorizer.pkl (TF-IDF)
```

**Solution:** Make sure you're pointing to the correct model directory that was created during training.

## Output Files

### Evaluation Results (`evaluation.json`)

```json
{
  "model_dir": "experiments/mlp_word2vec_full",
  "model_type": "word2vec",  // ← NEW: "word2vec" or "tfidf"
  "test_set_size": 3000,
  "num_predictions": 3000,
  "metrics": {
    "mae": 0.0530,
    "rmse": 0.0757,
    "correlation": 0.4567,
    "r2_score": 0.1234,
    "mape": 11.23
  },
  "analysis": {
    "worst_predictions": [...],
    "best_predictions": [...],
    "statistics": {...}
  }
}
```

### Plots (`plots/`)

- `predictions_vs_ground_truth.png` - Scatter plot
- `error_distribution.png` - Signed error histogram
- `absolute_error_distribution.png` - Absolute error histogram

## Comparison Examples

### Compare Word2Vec vs TF-IDF

```bash
# Evaluate both models
python scripts/evaluate_on_test_set.py \
  --model-dir experiments/mlp_word2vec_full \
  --output results/word2vec_eval.json \
  --plot-dir results/word2vec_plots

python scripts/evaluate_on_test_set.py \
  --model-dir experiments/mlp_deep \
  --output results/tfidf_eval.json \
  --plot-dir results/tfidf_plots

# Compare results
cat results/word2vec_eval.json | grep '"mae"'
cat results/tfidf_eval.json | grep '"mae"'
```

## Quick Test

```bash
# Quick test on a newly trained model
source .venv/bin/activate

# First train a model
python scripts/train_mlp_regressor.py \
  --limit 1000 \
  --epochs 10 \
  --output-dir experiments/test_model \
  --device cpu

# Then evaluate it
python scripts/evaluate_on_test_set.py \
  --model-dir experiments/test_model \
  --output results/test_eval.json \
  --device cpu
```

## What's Backwards Compatible

✅ **Old TF-IDF models still work** - No changes needed!  
✅ **Same command-line interface** - All arguments unchanged  
✅ **Same output format** - Just added `model_type` field  

## Summary of Files Modified

| File | Changes |
|------|---------|
| `scripts/evaluate_on_test_set.py` | • Auto-detect Word2Vec vs TF-IDF<br>• Add Word2Vec feature extraction<br>• Add StandardScaler support<br>• Enhanced loading output<br>• Added `model_type` to results |

## Troubleshooting

### Error: No module named 'gensim'

**Solution:** Install gensim
```bash
source .venv/bin/activate
uv pip install gensim
```

### Error: FileNotFoundError for word2vec_model.bin

**Cause:** Model wasn't trained with the new Word2Vec script

**Solution:** Retrain the model with the updated training script:
```bash
python scripts/train_mlp_regressor.py \
  --output-dir experiments/my_word2vec_model \
  --epochs 50
```

### Wrong predictions / Poor performance

**Cause:** Using TF-IDF model but it's loading as Word2Vec (or vice versa)

**Solution:** Check what files are in the model directory:
```bash
ls experiments/YOUR_MODEL_DIR/
```

Should contain either:
- Word2Vec: `word2vec_model.bin`, `feature_scaler.pkl`, `best_model.pt`
- TF-IDF: `tfidf_vectorizer.pkl`, `best_model.pt`

---

**Bottom Line:** The evaluation script now works with both TF-IDF and Word2Vec models automatically! 🎉

