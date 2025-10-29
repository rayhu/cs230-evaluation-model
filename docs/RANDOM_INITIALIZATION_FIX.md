# Random Initialization Fix

## Problem

The initial training loss was **too low** (0.2675), suggesting that even with "random" weights, the model was making good predictions. This indicates:

1. Word2Vec features were too predictive on their own
2. Default initialization wasn't random enough
3. No input standardization to normalize feature scales

## Solution: Truly Random Initialization

### 1. High-Variance Weight Initialization

**Before** (Kaiming/He init):
```python
nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
nn.init.constant_(m.bias, 0)  # Zeros - not random!
```

**After** (High variance random):
```python
# Random normal with high standard deviation
std = 0.5  # Much higher than typical ~0.01-0.1
nn.init.normal_(m.weight, mean=0.0, std=std)

# Random bias (not zeros!)
nn.init.uniform_(m.bias, -0.5, 0.5)
```

**Why this helps:**
- Higher std (0.5) creates more random initial predictions
- Random biases prevent systematic bias
- Ensures initial loss is higher (predictions are random)

### 2. Random BatchNorm Initialization

**Before**:
```python
nn.init.constant_(m.weight, 1)  # Always 1
nn.init.constant_(m.bias, 0)    # Always 0
```

**After**:
```python
# Random BatchNorm scaling
nn.init.uniform_(m.weight, 0.5, 1.5)  # Random scale
nn.init.uniform_(m.bias, -0.1, 0.1)   # Random shift
```

**Why this helps:**
- Prevents BatchNorm from having predictable behavior initially
- Adds more randomness to the forward pass

### 3. Input Feature Standardization

**New Addition**:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Zero mean, unit variance
X_val = scaler.transform(X_val)
```

**Why this helps:**
- Prevents Word2Vec features from dominating due to scale
- Ensures all features contribute equally initially
- Random weights have equal chance to affect all features

### 4. Input Dropout

**New Layer**:
```python
nn.Dropout(dropout_rate * 0.5),  # Dropout on input features
nn.Linear(input_dim, hidden_dim1),
...
```

**Why this helps:**
- Randomly drops 15% of input features (with dropout=0.3)
- Forces model to not rely on any specific features initially
- Increases randomness in predictions

### 5. Explicit Random Seed Control

**Added**:
```python
import random
import time

# Support for random seed each run
if args.seed == -1:
    args.seed = int(time.time()) % 100000

# Set all random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
```

**Why this helps:**
- Ensures reproducibility when needed
- Allows truly random runs with `--seed -1`
- Controls all sources of randomness

### 6. Initialization Verification

**Added Check**:
```python
# Verify random initialization
model.eval()
with torch.no_grad():
    sample_input = torch.randn(1, input_dim)
    sample_output = model(sample_input)
    print(f"Sample random prediction: {sample_output.item():.4f}")
```

**Why this helps:**
- Confirms weights are initialized randomly
- Shows unpredictable initial predictions
- Debugging tool for initialization issues

## Expected Results

### Before (Too Low):
```
Epoch 1: Train Loss = 0.2675  ← Too predictive!
```

### After (Higher, Random):
```
Expected Epoch 1: Train Loss = 0.5 - 1.5  ← Random predictions
```

With truly random initialization:
- **Initial loss should be much higher** (0.5-1.5 range)
- Predictions should be unpredictable before training
- Model should learn gradually, not start "good"

## How to Use

### Standard Training (Reproducible):
```bash
python scripts/train_mlp_regressor.py \
  --seed 42 \
  --output-dir experiments/run1
```

### Random Training (Different each time):
```bash
python scripts/train_mlp_regressor.py \
  --seed -1 \
  --output-dir experiments/run_random
```

### Test Random Initialization:
```bash
# Small test to verify high initial loss
python scripts/train_mlp_regressor.py \
  --limit 100 \
  --epochs 3 \
  --seed -1
```

## Technical Details

### Weight Initialization Variance

The standard deviation of 0.5 is chosen because:

1. **Input dimension**: 300 (Word2Vec)
2. **Typical Kaiming std**: `sqrt(2/fan_in) ≈ sqrt(2/300) ≈ 0.08`
3. **Our std**: 0.5 (6x higher than typical)

This creates much more variance in initial predictions:
```
Typical init: weights ~ N(0, 0.08²) = N(0, 0.0064)
Our init:     weights ~ N(0, 0.5²)  = N(0, 0.25)
```

Result: **39x more variance** in weight distribution!

### Why Not Even Higher?

We could use std=1.0 or higher, but:
- Too high (>1.0) can cause gradient explosion
- BatchNorm will stabilize during training
- 0.5 is a good balance: random but stable

## Verification Checklist

✅ **High-variance weight init** (std=0.5)  
✅ **Random bias init** (not zeros)  
✅ **Random BatchNorm init**  
✅ **Input standardization** (StandardScaler)  
✅ **Input dropout** (15%)  
✅ **Seed control** (reproducible + random modes)  
✅ **Verification check** (sample prediction)  

## Files Modified

| File | Changes |
|------|---------|
| `src/mlp_regressor.py` | • High-variance random init (std=0.5)<br>• Random BatchNorm init<br>• Input dropout layer |
| `scripts/train_mlp_regressor.py` | • StandardScaler normalization<br>• Explicit seed control<br>• Verification check<br>• Random seed option (`--seed -1`) |

## Summary

The combination of:
1. ✅ High-variance weight initialization (std=0.5)
2. ✅ Random bias initialization
3. ✅ Input feature standardization
4. ✅ Input dropout (15%)
5. ✅ Random BatchNorm initialization

Should result in **truly random initial predictions** with:
- **Initial loss**: 0.5 - 1.5 (instead of 0.27)
- **Random predictions** before training
- **Gradual learning** (not starting near-optimal)

The model will now properly learn from scratch instead of having suspiciously good initial predictions!

