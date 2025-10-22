# Table Extraction Evaluation Guide

## Overview

This guide explains how to evaluate table extraction quality by comparing extracted JSON files with ground truth annotations from the SciTSR dataset.

## Evaluation Metrics

### 1. Cell Detection Metrics (IoU-based)

**How it works:**
- Each predicted cell is matched with ground truth cells using **IoU (Intersection over Union)** of grid positions
- A match is successful if IoU ≥ threshold (default: 0.5)
- Calculates: Precision, Recall, F1 Score

**Formula:**
```
IoU = |intersection of grid cells| / |union of grid cells|

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Example:**
- Predicted cell spans: rows [0-2], cols [0-1] → occupies 9 grid positions
- Ground truth cell spans: rows [0-1], cols [0-1] → occupies 4 grid positions
- Intersection: 4 positions, Union: 9 positions
- IoU = 4/9 = 0.44 (below 0.5 threshold, no match)

### 2. Content Accuracy

**Text Similarity (Word-level Jaccard):**
- Compares text content of matched cells
- Formula: `|common words| / |total unique words|`
- Example: 
  - Predicted: "# settled nodes" → {#, settled, nodes}
  - Ground truth: "# settled nodes" → {#, settled, nodes}
  - Similarity: 3/3 = 1.0 (100%)

**Exact Match Rate:**
- Percentage of matched cells with identical text (case-insensitive, whitespace-normalized)

### 3. Structure Accuracy

**Row/Column Accuracy:**
- Measures how well the table dimensions are detected
- Formula: `1 - |predicted - actual| / max(predicted, actual)`
- Example:
  - Predicted: 3 rows, Actual: 4 rows
  - Row Accuracy: 1 - |3-4|/4 = 0.75 (75%)

### 4. Overall Score

Weighted combination of all metrics:
```
Overall Score = 
  0.40 × F1 Score +            # Cell detection (most important)
  0.30 × Text Similarity +      # Content accuracy
  0.15 × Row Accuracy +         # Structure
  0.15 × Col Accuracy
```

## Using the Scoring Script

### Single File Comparison

Compare one extraction output with its ground truth:

```bash
python scripts/score_extraction.py \
  --pred data/SciTSR/test/json_output/0704.1068v2.1.json \
  --gt data/SciTSR/test/structure/0704.1068v2.1.json \
  --detailed
```

**Output:**
```
📊 OVERALL SCORE: 0.225 (22.5%)

🔍 Cell Detection:
  Precision: 0.000 (0.0%)
  Recall:    0.000 (0.0%)
  F1 Score:  0.000 (0.0%)
  TP: 0, FP: 7, FN: 12

📝 Content Accuracy:
  Avg Text Similarity: 0.000 (0.0%)
  Exact Match Rate:    0.000 (0.0%)

🏗️  Structure Accuracy:
  Row Accuracy: 0.750 (75.0%)
  Col Accuracy: 0.750 (75.0%)
```

### Batch Evaluation

Evaluate all files in a directory:

```bash
python scripts/score_extraction.py \
  --pred data/SciTSR/test/json_output \
  --gt data/SciTSR/test/structure \
  --output results/evaluation_scores.json
```

**Output:**
```
📊 Average Scores:
  Overall Score:     0.456 ± 0.123
  Cell F1:           0.523 ± 0.187
  Cell Precision:    0.612 ± 0.152
  Cell Recall:       0.467 ± 0.201
  Text Similarity:   0.723 ± 0.145
  ...

📈 Score Distribution:
  0.0-0.2:   45 ( 1.5%) █
  0.2-0.4:  234 ( 7.8%) ████
  0.4-0.6:  892 (29.7%) ███████████████
  0.6-0.8: 1456 (48.5%) ████████████████████████
  0.8-1.0:  373 (12.4%) ██████
```

### Adjusting IoU Threshold

The IoU threshold controls how strict cell matching is:

```bash
# Strict matching (default)
--iou-threshold 0.5

# Lenient matching (for merged cells)
--iou-threshold 0.3

# Very strict
--iou-threshold 0.7
```

## Interpreting Scores

### Excellent (0.8-1.0)
- Near-perfect cell detection
- High content accuracy
- Correct table structure
- Very few errors

### Good (0.6-0.8)
- Most cells detected correctly
- Minor boundary errors
- Good text recognition
- Acceptable for many applications

### Fair (0.4-0.6)
- Significant cell detection issues
- Some merged or split cells
- Moderate text errors
- May need manual correction

### Poor (0.2-0.4)
- Many cells missed or incorrectly detected
- Major structural problems
- Low text accuracy
- Significant extraction failures

### Failed (0.0-0.2)
- Table structure not recognized
- Most cells incorrect
- Unusable extraction
- Complete failure

## Example Analysis

### Case 1: Good Extraction (Score: 0.82)
```json
{
  "cell_detection": {"f1": 0.91, "precision": 0.89, "recall": 0.93},
  "content_accuracy": {"avg_text_similarity": 0.95},
  "structure_accuracy": {"row_accuracy": 1.0, "col_accuracy": 1.0}
}
```
**Analysis:** Excellent performance. Almost all cells detected with correct boundaries and text.

### Case 2: Partial Success (Score: 0.54)
```json
{
  "cell_detection": {"f1": 0.67, "precision": 0.71, "recall": 0.63},
  "content_accuracy": {"avg_text_similarity": 0.78},
  "structure_accuracy": {"row_accuracy": 0.88, "col_accuracy": 0.75}
}
```
**Analysis:** Decent detection but some cells merged/split. Structure mostly correct. Usable with some cleanup.

### Case 3: Poor Extraction (Score: 0.23)
```json
{
  "cell_detection": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
  "content_accuracy": {"avg_text_similarity": 0.0},
  "structure_accuracy": {"row_accuracy": 0.75, "col_accuracy": 0.75}
}
```
**Analysis:** Cell boundaries completely wrong (likely merged). Structure dimensions close but cells don't match. Extraction failed.

## Common Error Patterns

### 1. Merged Cells
**Symptom:** Low recall, detected cells span multiple ground truth cells
```
Predicted: 1 large cell spanning rows 0-2, cols 0-3
Ground Truth: 12 individual cells in 3×4 grid
Result: 0% F1, poor overall score
```

### 2. Over-segmentation
**Symptom:** Low precision, too many small cells detected
```
Predicted: 20 small cells
Ground Truth: 10 cells
Result: High FP count, low precision
```

### 3. Boundary Errors
**Symptom:** Moderate F1, cells detected but with wrong spans
```
Predicted cell: rows [0-1], cols [0-0]
Ground Truth: rows [0-0], cols [0-0]
IoU: 0.5 (borderline match)
```

### 4. Text Errors
**Symptom:** Good F1 but low text similarity
```
Cell matched but OCR errors:
Predicted: "18965" (OCR error)
Ground Truth: "18966"
Text similarity: 0.0 (no word overlap)
```

## Using Scores for Training

### For Your Neural Verifier

1. **Input Features:**
   - Extracted table JSON
   - Table image
   - Predicted cell boundaries
   - OCR confidence scores

2. **Target Labels:**
   - Overall quality score (0-1)
   - Cell detection F1
   - Content accuracy
   - Structure correctness

3. **Training Data:**
   - Generate scores for all 3000 test tables
   - Create pairs: (extraction, ground_truth, score)
   - Model learns to predict quality without ground truth

4. **Evaluation:**
   - Model should predict low scores for poor extractions
   - High scores for good extractions
   - Correlation with actual metrics

## Advanced Usage

### Save Detailed Results

```bash
python scripts/score_extraction.py \
  --pred data/SciTSR/test/json_output \
  --gt data/SciTSR/test/structure \
  --output results/detailed_scores.json
```

The output JSON contains:
- Individual file scores
- Aggregated statistics
- Score distributions
- Per-file detailed metrics

### Programmatic Usage

```python
from scripts.score_extraction import compare_files

scores = compare_files(
    pred_file=Path("output.json"),
    gt_file=Path("ground_truth.json"),
    iou_threshold=0.5
)

print(f"Overall score: {scores['overall_score']}")
print(f"F1: {scores['cell_detection']['f1']}")
```

## References

- **IoU (Intersection over Union):** Standard metric for object detection
- **Jaccard Similarity:** Set-based text similarity measure
- **TEDS:** Tree-Edit-Distance-based Similarity (used in industry)
- **SciTSR Dataset:** https://github.com/Academic-Hammer/SciTSR

## Next Steps

1. **Process all 3000 test images:**
   ```bash
   python scripts/extract_tables_scitsr.py \
     --input data/SciTSR/test/img \
     --output data/SciTSR/test/json_output
   ```

2. **Generate evaluation scores:**
   ```bash
   python scripts/score_extraction.py \
     --pred data/SciTSR/test/json_output \
     --gt data/SciTSR/test/structure \
     --output results/all_scores.json
   ```

3. **Analyze results and build your neural verifier:**
   - Load scores and features
   - Train model to predict quality
   - Evaluate on held-out set

