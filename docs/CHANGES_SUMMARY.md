# Summary of Changes: TF-IDF to Word2Vec Migration

## Quick Comparison

| Aspect | Before (TF-IDF) | After (Word2Vec) |
|--------|-----------------|------------------|
| **Feature Extraction** | Sparse TF-IDF vectors | Dense Word2Vec embeddings |
| **Feature Dimension** | 10,000 (configurable) | 300 (configurable) |
| **Initial Loss** | ~0.017 (too low) | Expected ~0.1-0.3 (healthy) |
| **Batch Normalization** | ❌ No | ✅ Yes |
| **Weight Initialization** | Default (random) | ✅ Kaiming/He for ReLU |
| **Default Dropout** | 0.0 (none) | 0.3 (regularized) |
| **Semantic Understanding** | ❌ No | ✅ Yes |
| **Training Time** | ~2-3 min | ~5-10 min |

## What Was Wrong Before?

Looking at your training history:
```json
{
  "train_loss": [
    0.01738303018656249,  // Epoch 1 - Already very low!
    0.006792802804460128, // Epoch 2 - Dropping fast
    0.004690087265024,    // Epoch 3 - Suspiciously good
    ...
  ]
}
```

**Problems:**
1. **Initial loss too low** (0.017) - suggests the model was overfitting immediately
2. **Rapid decrease** - no proper learning curve, just memorization
3. **No regularization** - dropout=0, no batch norm
4. **Poor initialization** - default PyTorch init not optimal for this task

## What Changed?

### 1. Feature Extraction: TF-IDF → Word2Vec

**Old Code:**
```python
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(texts_train).toarray()
```

**New Code:**
```python
# Tokenize JSON text
train_tokens = [tokenize_json(text) for text in texts_train]

# Train Word2Vec
w2v_model = Word2Vec(
    sentences=train_tokens,
    vector_size=300,
    window=5,
    sg=1,  # Skip-gram
    epochs=10
)

# Average word vectors for each document
X_train = np.array([doc_to_vec(tokens, w2v_model) for tokens in train_tokens])
```

### 2. Model Architecture: Added Regularization

**Old Architecture:**
```python
layers = [
    nn.Linear(input_dim, hidden_dim1),
    nn.ReLU(),
    nn.Dropout(dropout_rate),  # dropout_rate=0 by default!
    nn.Linear(hidden_dim1, hidden_dim2),
    nn.ReLU(),
    nn.Dropout(dropout_rate),
    nn.Linear(hidden_dim2, 1)
]
```

**New Architecture:**
```python
layers = [
    nn.Linear(input_dim, hidden_dim1),
    nn.BatchNorm1d(hidden_dim1),      # ← NEW
    nn.ReLU(),
    nn.Dropout(0.3),                   # ← INCREASED from 0
    nn.Linear(hidden_dim1, hidden_dim2),
    nn.BatchNorm1d(hidden_dim2),      # ← NEW
    nn.ReLU(),
    nn.Dropout(0.3),                   # ← INCREASED from 0
    nn.Linear(hidden_dim2, 1)
]

# ← NEW: Proper weight initialization
self._initialize_weights()
```

### 3. Weight Initialization: Random → Kaiming

**New Method:**
```python
def _initialize_weights(self):
    """Initialize weights with He initialization for ReLU activations."""
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
```

## How to Train Now

### Old Command:
```bash
python scripts/train_mlp_regressor.py \
  --max-features 10000 \
  --dropout 0.0
```

### New Command:
```bash
python scripts/train_mlp_regressor.py \
  --vector-size 300 \
  --window 5 \
  --dropout 0.3
```

## Expected Training Behavior

### Before (Your Current Results):
```
Epoch 1/50
  Train Loss: 0.0174  ← Too low!
  Val MAE:    0.0673
  Val RMSE:   0.0933

Epoch 2/50
  Train Loss: 0.0068  ← Dropping too fast!
  Val MAE:    0.0677  ← Getting worse!
  Val RMSE:   0.0941
```

### After (Expected with New Setup):
```
Epoch 1/50
  Train Loss: 0.15-0.25  ← Healthier starting point
  Val MAE:    0.10-0.12
  Val RMSE:   0.14-0.16

Epoch 5/50
  Train Loss: 0.08-0.12  ← Gradual improvement
  Val MAE:    0.08-0.09  ← Actually improving!
  Val RMSE:   0.11-0.13

Epoch 20/50
  Train Loss: 0.05-0.07  ← Steady convergence
  Val MAE:    0.07-0.08  ← Better generalization
  Val RMSE:   0.09-0.11
```

## Files Modified

✅ `src/mlp_regressor.py` - Added BatchNorm + weight init  
✅ `scripts/train_mlp_regressor.py` - Word2Vec instead of TF-IDF  
✅ `requirements.txt` - Added gensim  
✅ `docs/WORD2VEC_MIGRATION.md` - Full documentation  

## Quick Start

```bash
# 1. Install new dependency
pip install gensim>=4.3.0

# 2. Train a new model
python scripts/train_mlp_regressor.py \
  --output-dir experiments/mlp_word2vec_test \
  --epochs 50 \
  --device mps  # or cuda/cpu

# 3. Quick test with limited data
python scripts/train_mlp_regressor.py \
  --limit 1000 \
  --epochs 10 \
  --output-dir experiments/test_run
```

## Why This Is Better

✅ **Proper Initialization** - Kaiming/He init prevents overfitting from the start  
✅ **Better Features** - Word2Vec captures semantic relationships  
✅ **Regularization** - BatchNorm + Dropout (0.3) prevents overfitting  
✅ **Realistic Loss** - Initial loss will be higher but will generalize better  
✅ **Stable Training** - BatchNorm stabilizes gradient flow  

## Next Steps

1. **Retrain your model** with the new architecture
2. **Watch the training curves** - they should look much healthier
3. **Compare performance** on held-out test data
4. **Experiment** with hyperparameters if needed

See `docs/WORD2VEC_MIGRATION.md` for detailed documentation.

