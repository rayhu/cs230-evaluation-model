# Data Augmentation for Table Extraction Evaluation Dataset

## Problem

The dataset has severe class imbalance:
- **76.8%** of samples are in the 0.4-0.6 (Medium) range
- Only **0.3%** in Very Low (0.0-0.2) range
- Only **0.1%** in Very High (0.8-1.0) range

This imbalance can cause the model to:
- Overfit to the medium range
- Poorly predict extreme scores
- Have reduced generalization

## Solution

The augmentation script creates synthetic variations of existing tables to balance the distribution by:
1. **Structural modifications**: Shifting, merging, splitting, or removing cells
2. **Oversampling**: Generating more samples for underrepresented ranges
3. **Controlled generation**: Creating variants that target specific score ranges

## Usage

### Step 1: Analyze Current Distribution

```bash
python scripts/analyze_score_distribution.py \
    --metadata dataset/train/metadata_train.jsonl
```

Optional: Generate a plot:
```bash
python scripts/analyze_score_distribution.py \
    --metadata dataset/train/metadata_train.jsonl \
    --plot distribution_before.png
```

### Step 2: Augment the Dataset

**Basic usage** (balance to current maximum):
```bash
python scripts/augment_dataset.py \
    --metadata dataset/train/metadata_train.jsonl \
    --generated dataset/train/generated \
    --ground-truth dataset/train/ground_truth \
    --output dataset/train/metadata_train_augmented.jsonl \
    --augmentation-factor 1.0
```

**Custom target distribution**:
```bash
python scripts/augment_dataset.py \
    --metadata dataset/train/metadata_train.jsonl \
    --generated dataset/train/generated \
    --ground-truth dataset/train/ground_truth \
    --output dataset/train/metadata_train_augmented.jsonl \
    --target-very-low 500 \
    --target-low 2000 \
    --target-medium 3000 \
    --target-high 2000 \
    --target-very-high 500
```

**Double the dataset size**:
```bash
python scripts/augment_dataset.py \
    --metadata dataset/train/metadata_train.jsonl \
    --generated dataset/train/generated \
    --ground-truth dataset/train/ground_truth \
    --output dataset/train/metadata_train_augmented.jsonl \
    --augmentation-factor 2.0
```

### Step 3: Verify Augmented Distribution

```bash
python scripts/analyze_score_distribution.py \
    --metadata dataset/train/metadata_train_augmented.jsonl
```

## Augmentation Strategies

### 1. Low Score Variants (0.0-0.4)
- **Shift cells**: Randomly move cell positions
- **Remove cells**: Delete some cells (up to 50%)
- **Merge cells**: Combine adjacent cells incorrectly
- **Multiple modifications**: Apply 2-4 modifications

### 2. High Score Variants (0.6-1.0)
- **Minimal modifications**: Start from ground truth
- **Small shifts**: Minor position adjustments
- **Preserve structure**: Keep most of the original layout

### 3. Medium Score Variants (0.4-0.6)
- **Moderate modifications**: Mix of shifts and merges
- **Controlled changes**: Keep similarity in target range

## Parameters

- `--augmentation-factor`: Multiplier for augmentation (default: 1.0)
  - 1.0 = balance to current maximum count
  - 2.0 = double the maximum count
  - 0.5 = half the maximum count

- `--target-very-low`, `--target-low`, etc.: Explicit target counts for each range

## Output Format

Augmented samples include:
- Original metadata fields: `id`, `ground_truth_file`, `generated_file`, `similarity_score`
- New fields:
  - `augmented_from`: ID of the original sample
  - `augmentation_type`: `'low'`, `'medium'`, or `'high'`

## Example Output

```json
{
  "id": "0001020v1.11_aug_low_1234",
  "ground_truth_file": "0001020v1.11.json",
  "generated_file": "0001020v1.11_aug_low_1234.json",
  "similarity_score": 0.32,
  "augmented_from": "0001020v1.11",
  "augmentation_type": "low"
}
```

## Notes

1. **Ground truth files are not modified** - only generated files are augmented
2. **New JSON files are created** in the same directory as original generated files
3. **Original samples are preserved** - augmentation adds new samples, doesn't replace
4. **Score calculation** uses the same `score_extraction.py` evaluation function

## Performance Considerations

- Augmentation can be slow for large datasets
- Each augmented sample requires:
  - Loading original files
  - Applying modifications
  - Calculating similarity score
  - Saving new files

For 10,000 samples, expect ~30-60 minutes depending on hardware.

## Troubleshooting

**Issue**: "Error augmenting sample"
- Check that all ground truth and generated files exist
- Verify JSON file format is correct
- Some tables may be too small to augment

**Issue**: Generated scores not in target range
- The augmentation tries to hit target ranges but may not always succeed
- Try running with more samples or adjust target ranges

**Issue**: Too many/few samples generated
- Adjust `--augmentation-factor` or use explicit `--target-*` parameters
- Check the final distribution analysis

## Next Steps

After augmentation:
1. Train your model on the augmented dataset
2. Compare performance with original dataset
3. Monitor if the model better handles extreme scores
4. Consider using weighted loss functions as an alternative/additional approach

