# How to Visualize Dataset Distribution

This guide explains how to visualize the score distribution of your dataset using the `analyze_score_distribution.py` script.

## Quick Start

### For JSONL Files

```bash
# Activate virtual environment
source .venv/bin/activate

# Analyze a JSONL metadata file
python scripts/analyze_score_distribution.py \
  --metadata dataset/train/metadata_train_augmented_high_v3.jsonl \
  --plot results/distribution_v3.png
```

### For Parquet Datasets

```bash
# Analyze local Parquet dataset
python scripts/analyze_score_distribution.py \
  --parquet dataset_parquet_v3 \
  --split train \
  --plot results/distribution_v3.png

# Analyze from Hugging Face
python scripts/analyze_score_distribution.py \
  --parquet rayhu/table-extraction-evaluation \
  --split train \
  --plot results/distribution_hf.png
```

## Available Datasets

### JSONL Files (in `dataset/train/`)
- `metadata_train.jsonl` - Original training dataset
- `metadata_train_augmented_high.jsonl` - Version 2 with high bucket augmentation
- `metadata_train_augmented_high_v3.jsonl` - Version 3 with 9,481 high bucket samples

### Parquet Datasets
- `dataset_parquet_v3/` - Local version 3 dataset (36,066 samples)
- `rayhu/table-extraction-evaluation` - Hugging Face dataset

## Command Options

```
--metadata PATH          Path to JSONL metadata file
--parquet PATH/ID        Path to local parquet directory or Hugging Face dataset ID
--split SPLIT            Dataset split to analyze (default: train)
--plot PATH              Save visualization plot to file (optional)
```

## Output

The script provides:

1. **Console Output**: 
   - Total samples
   - Mean, median, standard deviation
   - Score range
   - Distribution by buckets (Very Low, Low, Medium, High, Very High)

2. **Visualization** (if `--plot` is provided):
   - Histogram showing score distribution
   - Bar chart showing bucket counts and percentages

## Examples

### Example 1: View Original Dataset Distribution

```bash
python scripts/analyze_score_distribution.py \
  --metadata dataset/train/metadata_train.jsonl
```

### Example 2: Compare Version 3 Distribution

```bash
python scripts/analyze_score_distribution.py \
  --metadata dataset/train/metadata_train_augmented_high_v3.jsonl \
  --plot results/distribution_v3.png
```

### Example 3: Analyze Parquet Version 3

```bash
python scripts/analyze_score_distribution.py \
  --parquet dataset_parquet_v3 \
  --split train \
  --plot results/distribution_parquet_v3.png
```

### Example 4: Analyze Test Set

```bash
python scripts/analyze_score_distribution.py \
  --parquet dataset_parquet_v3 \
  --split test \
  --plot results/distribution_test.png
```

## Understanding the Output

### Score Buckets
- **Very Low (0.0-0.2)**: Poor quality extractions
- **Low (0.2-0.4)**: Below average quality
- **Medium (0.4-0.6)**: Average quality (most common)
- **High (0.6-0.8)**: Good quality (target for augmentation)
- **Very High (0.8-1.0)**: Excellent quality

### Key Metrics
- **Mean**: Average similarity score
- **Median**: Middle value (less affected by outliers)
- **Std Deviation**: Measure of spread
- **Range**: Min to max score values

## Tips

1. **Compare Distributions**: Run the script on different versions to see how augmentation affects the distribution
2. **Save Plots**: Use `--plot` to save visualizations for reports
3. **Test Set Analysis**: Always check test set distribution to ensure it matches training distribution
4. **Bucket Balance**: Aim for balanced distribution across buckets for better model training

## Troubleshooting

### Error: "matplotlib not available"
```bash
pip install matplotlib
```

### Error: "datasets library is required"
```bash
pip install datasets
```

### Error: "Metadata file does not exist"
- Check the file path is correct
- Ensure you're in the project root directory

### Error: "Could not load dataset"
- For local parquet: Ensure the directory exists and contains parquet files
- For Hugging Face: Check your internet connection and dataset ID

