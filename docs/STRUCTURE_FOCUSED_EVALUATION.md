# Structure-Focused Evaluation for Table Extraction

## Overview

The evaluation system has been updated to focus on **table structure accuracy** rather than cell content accuracy. This is ideal for automated evaluation models that need to assess extraction quality without ground truth content.

## Changes Made

### Weight Distribution

**Previous (Mixed Focus):**
```python
overall_score = (
    0.40 × F1 +                  # Cell detection
    0.30 × Text Similarity +     # Content accuracy (REMOVED)
    0.15 × Row Accuracy +
    0.15 × Column Accuracy
)
```

**Current (Structure-Focused):**
```python
overall_score = (
    0.50 × F1 +                  # Cell detection (higher weight)
    0.25 × Row Accuracy +        # Grid structure
    0.25 × Column Accuracy       # Grid structure
)
```

## Rationale

### Why Structure-Only?

1. **OCR Independence**: Text quality depends on OCR models, not table structure understanding
2. **Generalization**: Structure patterns work across all table types
3. **Ground Truth Requirement**: Content accuracy requires manual annotation
4. **Objective Assessment**: Structure correctness is unambiguous

### Example

Your test case (`0704.1068v2.1.json`):
- **Structure**: 3×4 grid detected vs actual 4×3 = 75% accuracy ✅
- **Cell Boundaries**: All cells merged = 0% F1 ❌
- **Old Score**: 22.5% (penalized by content)
- **New Score**: 37.5% (better reflects structure capability)

## Metrics Breakdown

### 1. Cell Detection (50% weight)

Measures how accurately individual cells are detected:
- Uses IoU-based matching on grid positions
- Indicates if cells are correctly split vs merged
- Critical for structure evaluation

**Your Result**: 0% (all cells merged incorrectly)

### 2. Row Accuracy (25% weight)

Measures grid dimension detection:
```
Accuracy = 1 - |predicted_rows - actual_rows| / max(predicted, actual)
```

**Your Result**: 75% (detected 3 rows vs actual 4 rows)

### 3. Column Accuracy (25% weight)

Measures grid dimension detection:
```
Accuracy = 1 - |predicted_cols - actual_cols| / max(predicted, actual)
```

**Your Result**: 75% (detected 4 cols vs actual 3 cols)

## Using the Updated System

### Single File Evaluation

```bash
python scripts/score_extraction.py \
  --pred data/SciTSR/test/json_output/0704.1068v2.1.json \
  --gt data/SciTSR/test/structure/0704.1068v2.1.json
```

### Batch Evaluation

```bash
python scripts/score_extraction.py \
  --pred data/SciTSR/test/json_output \
  --gt data/SciTSR/test/structure \
  --output results/scores.json
```

**Output Now Shows:**
```
📊 OVERALL SCORE: 0.375 (37.5%) (structure-focused)

🔍 Cell Detection:
  F1 Score: 0.000 (0.0%)

🏗️  Structure Accuracy:
  Row Accuracy: 0.750 (75.0%)
  Col Accuracy: 0.750 (75.0%)

# Content accuracy section still shown but not included in score
```

## Impact on Your Neural Verifier

### Training Targets

Your evaluation model will learn to predict:
- **Structure correctness** (0-1 scale)
- Based on visual and structural features
- Independent of OCR quality

### Expected Training Data Distribution

With structure-focused scoring:
- Average scores likely in 0.3-0.6 range
- Reflects table complexity and extraction challenges
- More stable than content-dependent scoring

## Advantages

✅ **OCR Robustness**: Works regardless of text quality
✅ **Generalization**: Applicable to all table types  
✅ **Objective**: Unambiguous evaluation criteria
✅ **Practical**: Focuses on what matters for downstream tasks
✅ **Ground Truth**: No manual content annotation needed

## Content Accuracy Still Available

While removed from overall score, content metrics are still calculated for detailed analysis:
- Text similarity (available in individual scores)
- Exact match rate (available in individual scores)
- Useful for debugging OCR issues

Access via:
```python
scores['content_accuracy']['avg_text_similarity']
scores['content_accuracy']['exact_match_rate']
```

## References

- Cell Detection F1: Industry standard for table evaluation
- Grid Accuracy: Used in TEDS (Tree-Edit-Distance Score)
- Structure Focus: Aligns with GriTS (Grid Table Similarity) metric
