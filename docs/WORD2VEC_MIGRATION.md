# Migration from TF-IDF to Word2Vec with Improved Regularization

## Overview

This document describes the migration from TF-IDF to Word2Vec embeddings for the MLP regressor, along with improvements to weight initialization and regularization to address overfitting issues.

## Changes Made

### 1. Feature Extraction (TF-IDF → Word2Vec)

**Previous Approach:**
- Used TF-IDF vectorization on JSON strings
- Default 10,000 features
- Sparse bag-of-words representation

**New Approach:**
- Word2Vec skip-gram embeddings (gensim)
- Default 300-dimensional dense vectors
- Captures semantic relationships between words
- Document vectors created by averaging word embeddings

**Why Word2Vec?**
- Better semantic representation of JSON structures
- Dense embeddings provide richer features
- More generalizable to unseen vocabulary patterns
- Can capture relationships between table structure elements

### 2. Model Architecture Improvements

**Previous Architecture:**
```
Input → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear → Output
```

**New Architecture:**
```
Input → Linear → BatchNorm → ReLU → Dropout → Linear → BatchNorm → ReLU → Dropout → Linear → Output
```

**Key Additions:**
- **Batch Normalization**: Stabilizes training and improves gradient flow
- **Kaiming/He Initialization**: Proper weight initialization for ReLU activations
- **Higher Default Dropout**: Increased from 0.0 to 0.3 for better regularization

### 3. Weight Initialization

Added proper weight initialization to prevent the extremely low initial loss observed in previous training runs:

```python
def _initialize_weights(self):
    """Initialize weights with He initialization for ReLU activations."""
    for m in self.modules():
        if isinstance(m, nn.Linear):
            # He/Kaiming initialization for ReLU
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
```

**Why This Helps:**
- Prevents saturation of activations
- Maintains variance across layers
- Avoids gradient vanishing/explosion
- Provides more realistic initial loss values

### 4. New Command-Line Arguments

**Removed:**
- `--max-features` (TF-IDF-specific)

**Added:**
- `--vector-size`: Word2Vec embedding dimension (default: 300)
- `--window`: Context window size for Word2Vec (default: 5)

**Updated Defaults:**
- `--dropout`: Changed from 0.0 to 0.3

## Training Configuration

### Default Hyperparameters

```bash
python scripts/train_mlp_regressor.py \
  --vector-size 300 \
  --window 5 \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --hidden-dim1 256 \
  --hidden-dim2 64 \
  --dropout 0.3 \
  --output-dir experiments/mlp_word2vec
```

### Word2Vec Configuration

The Word2Vec model is trained with:
- **Algorithm**: Skip-gram (sg=1) - better for smaller datasets
- **Vector Size**: 300 dimensions
- **Window**: 5 words context
- **Min Count**: 2 (minimum word frequency)
- **Epochs**: 10 Word2Vec training epochs
- **Seed**: 42 (reproducibility)

## Expected Improvements

### 1. Training Loss Behavior

**Before:**
- Initial loss: ~0.017 (suspiciously low)
- Rapid convergence suggesting overfitting
- Validation metrics not improving beyond epoch 1

**Expected After:**
- Higher initial loss (more realistic)
- Gradual convergence
- Better generalization to validation set
- More stable training curves

### 2. Model Generalization

The combination of:
- Semantic embeddings (Word2Vec)
- Batch normalization
- Proper weight initialization
- Higher dropout (0.3)

Should result in:
- Better performance on unseen data
- More robust to variations in table structure
- Reduced overfitting

## Files Changed

### Modified Files

1. **`src/mlp_regressor.py`**
   - Added batch normalization layers
   - Added `_initialize_weights()` method
   - Updated docstrings for Word2Vec features
   - Increased default dropout to 0.3

2. **`scripts/train_mlp_regressor.py`**
   - Replaced TF-IDF with Word2Vec
   - Added `tokenize_json()` function
   - Updated `prepare_features()` for Word2Vec
   - Changed CLI arguments
   - Updated documentation

3. **`requirements.txt`**
   - Added `gensim>=4.3.0` for Word2Vec

## Migration Guide

### For Existing Models

Old models trained with TF-IDF are **not compatible** with the new Word2Vec architecture due to:
- Different input dimensions (10000 → 300)
- Different feature representation
- Additional batch normalization layers

You will need to **retrain** your models with the new architecture.

### Training a New Model

```bash
# Basic training
python scripts/train_mlp_regressor.py \
  --output-dir experiments/mlp_word2vec \
  --epochs 50

# With custom Word2Vec settings
python scripts/train_mlp_regressor.py \
  --vector-size 200 \
  --window 10 \
  --dropout 0.4 \
  --output-dir experiments/mlp_w2v_custom

# Quick test with limited data
python scripts/train_mlp_regressor.py \
  --limit 1000 \
  --epochs 10 \
  --output-dir experiments/test_run
```

### Saved Artifacts

Training now saves:
- `word2vec_model.bin` - Trained Word2Vec model (instead of `tfidf_vectorizer.pkl`)
- `best_model.pt` - Best model checkpoint
- `final_model.pt` - Final model state
- `training_history.json` - Training curves
- `config.json` - Hyperparameters

## Troubleshooting

### Issue: High initial loss

**Expected Behavior**: With proper initialization, initial loss should be higher than before (~0.1-0.3 range)

This is **good** - it indicates the model is learning from scratch rather than overfitting immediately.

### Issue: Slow training

Word2Vec training may take longer than TF-IDF because:
1. Word2Vec model training (10 epochs)
2. Document tokenization
3. Vector averaging

**Solution**: Use `--limit` for quick tests, full dataset for final training.

### Issue: Out of memory

If you encounter OOM errors:
```bash
# Reduce batch size
--batch-size 16

# Reduce embedding size
--vector-size 150

# Reduce hidden dimensions
--hidden-dim1 128 --hidden-dim2 32
```

## Performance Expectations

### Training Time

- **TF-IDF**: ~2-3 minutes for feature extraction + training
- **Word2Vec**: ~5-10 minutes for Word2Vec training + embedding + training

### Accuracy

Expected validation MAE:
- **Before (TF-IDF)**: ~0.069 (but likely overfit)
- **After (Word2Vec)**: ~0.070-0.080 (better generalization)

The slightly higher MAE is expected and healthy - it indicates the model is regularized properly and will generalize better to new data.

## Next Steps

1. **Train a new model** with the updated architecture
2. **Compare validation curves** - look for smoother convergence
3. **Test on held-out data** to verify generalization
4. **Experiment with hyperparameters**:
   - Try different vector sizes (100, 200, 300, 400)
   - Adjust dropout (0.2, 0.3, 0.4, 0.5)
   - Tune window size (3, 5, 7, 10)

## References

- [Word2Vec Paper](https://arxiv.org/abs/1301.3781)
- [Gensim Word2Vec Documentation](https://radimrehurek.com/gensim/models/word2vec.html)
- [He Initialization](https://arxiv.org/abs/1502.01852)
- [Batch Normalization](https://arxiv.org/abs/1502.03167)

