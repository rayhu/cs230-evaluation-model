# Table Extraction Quality Evaluation Model - Test Report

**Date**: December 2024  
**Model Version**: MLP Regressor v3  
**Dataset Version**: Version 3 (36,066 training samples)

---

## Executive Summary

This report presents the evaluation results of the MLP-based table extraction quality prediction model trained on Version 3 of the dataset. The model was trained on 36,066 samples with enhanced distribution in the high-quality bucket (0.6-0.8), achieving 9,481 samples (26.3% of the dataset) in this range. The model was evaluated on a held-out test set of 3,000 samples.

### Key Findings

- **Tolerance-based Accuracy**: The model achieves **78.9% accuracy** within ±10% tolerance and **89.8% accuracy** within ±15% tolerance
- **Mean Absolute Error**: 0.0648 (6.48 percentage points)
- **Median Absolute Error**: 0.0438 (4.38 percentage points)
- **Mean Absolute Percentage Error**: 15.05%
- **Correlation with Ground Truth**: 0.3655 (moderate)
- **R² Score**: -0.0792 (indicates room for improvement)

---

## 1. Model Architecture

### Architecture Details
- **Input Dimension**: 384 (Sentence Transformer embeddings)
- **Hidden Layer 1**: 512 units
- **Hidden Layer 2**: 256 units
- **Output**: 1 (quality score prediction)
- **Total Parameters**: ~400K

### Training Configuration
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Training Samples**: 32,459 (90% of 36,066)
- **Validation Samples**: 3,607 (10% of 36,066)
- **Batch Size**: 64
- **Learning Rate**: 0.0003 (with ReduceLROnPlateau scheduler)
- **Dropout Rate**: 0.1
- **Weight Decay (L2)**: 0.0001
- **Early Stopping Patience**: 15 epochs
- **Total Epochs Trained**: 99 (stopped early)

### Best Validation Performance
- **MAE**: 0.0528
- **RMSE**: 0.0762
- **R²**: 0.8648
- **MAPE**: 9.92%
- **Accuracy (±10%)**: 85.0%

---

## 2. Dataset Information

### Training Dataset (Version 3)
- **Total Samples**: 36,066
- **Distribution by Quality Buckets**:
  - Very Low (0.0-0.2): 67 samples (0.2%)
  - Low (0.2-0.4): 3,860 samples (10.7%)
  - Medium (0.4-0.6): 14,195 samples (39.4%)
  - **High (0.6-0.8): 9,481 samples (26.3%)** ⭐
  - Very High (0.8-1.0): 8,276 samples (22.9%)

- **Statistics**:
  - Mean Score: 0.620
  - Median Score: 0.587
  - Standard Deviation: 0.211
  - Score Range: 0.042 - 1.000

### Test Dataset
- **Total Samples**: 3,000
- **Statistics**:
  - Mean Score: 0.4578
  - Standard Deviation: 0.0884
  - Score Range: 0.1477 - 0.8158

**Note**: The test set has a lower mean score (0.46) compared to the training set (0.62), which may contribute to the model's performance challenges.

---

## 3. Test Set Evaluation Results

### 3.1 Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **MAE** | 0.0648 | Average prediction error of 6.48 percentage points |
| **Median AE** | 0.0438 | Median prediction error of 4.38 percentage points |
| **RMSE** | 0.0918 | Root mean squared error of 9.18 percentage points |
| **MAPE** | 15.05% | Mean absolute percentage error |
| **Correlation** | 0.3655 | Moderate positive correlation with ground truth |
| **R² Score** | -0.0792 | Negative R² indicates model performs worse than baseline |

### 3.2 Tolerance-Based Accuracy

The model's performance improves significantly when considering tolerance ranges:

| Tolerance | Accuracy | Interpretation |
|-----------|----------|---------------|
| ±1% (±0.01) | 13.2% | Exact predictions are rare |
| ±5% (±0.05) | 54.5% | More than half within 5 percentage points |
| **±10% (±0.10)** | **78.9%** | **Nearly 4 out of 5 predictions within 10 points** |
| ±15% (±0.15) | 89.8% | Nearly 9 out of 10 predictions within 15 points |

### 3.3 Prediction Distribution

**Predicted Scores**:
- Mean: 0.4708
- Standard Deviation: 0.0709
- Range: [0.1965, 0.9308]

**Ground Truth Scores**:
- Mean: 0.4578
- Standard Deviation: 0.0884
- Range: [0.1477, 0.8158]

**Analysis**: The model's predictions have a similar mean to ground truth (0.47 vs 0.46) but with lower variance (0.071 vs 0.088), suggesting the model is somewhat conservative in its predictions and may not capture the full range of quality scores.

---

## 4. Error Analysis

### 4.1 Best Predictions (Top 5)

| Rank | Sample ID | Predicted | Actual | Error |
|------|-----------|-----------|--------|-------|
| 1 | 1710.08180v1.2 | 0.4375 | 0.4375 | 0.0000 |
| 2 | 1610.07560v1.3 | 0.4687 | 0.4688 | 0.0000 |
| 3 | 1710.07695v1.4 | 0.4545 | 0.4545 | 0.0001 |
| 4 | 1504.05035v1.3 | 0.4584 | 0.4583 | 0.0001 |
| 5 | 1711.08229v3.4 | 0.4614 | 0.4613 | 0.0001 |

**Observation**: The model achieves near-perfect predictions for samples in the medium quality range (0.43-0.47), suggesting good performance for typical cases.

### 4.2 Worst Predictions (Top 5)

| Rank | Sample ID | Predicted | Actual | Error |
|------|-----------|-----------|--------|-------|
| 1 | 1504.07843v1.2 | 0.8062 | 0.3780 | 0.4282 |
| 2 | 1705.06920v5.9 | 0.6573 | 0.2360 | 0.4213 |
| 3 | 1803.01557v2.3 | 0.8500 | 0.4301 | 0.4199 |
| 4 | 1612.07833v1.1 | 0.7883 | 0.3824 | 0.4058 |
| 5 | 1712.10232v1.1 | 0.6728 | 0.2750 | 0.3978 |

**Observation**: The model consistently overestimates quality for low-quality samples (actual scores 0.24-0.43), predicting them as high-quality (0.66-0.85). This suggests the model struggles with distinguishing low-quality extractions.

---

## 5. Strengths and Limitations

### Strengths

1. **Tolerance-Based Performance**: The model achieves 78.9% accuracy within ±10% tolerance, which is acceptable for practical applications where exact precision is not required.

2. **Consistent Predictions**: The median absolute error (0.0438) is lower than the mean (0.0648), indicating most predictions are reasonably accurate with fewer extreme outliers.

3. **Good Coverage**: 89.8% of predictions fall within ±15% of ground truth, providing reliable estimates for most use cases.

4. **Balanced Mean Prediction**: The model's mean prediction (0.47) closely matches the test set mean (0.46), showing good calibration on average.

### Limitations

1. **Low Correlation**: The correlation coefficient of 0.3655 is moderate, suggesting the model may not capture all relevant patterns in the data.

2. **Negative R² Score**: The negative R² (-0.0792) indicates the model performs worse than a simple baseline (predicting the mean), which is concerning and suggests potential overfitting or distribution mismatch.

3. **Systematic Bias for Low-Quality Samples**: The model consistently overestimates quality for low-quality extractions, which could lead to false positives in quality assessment.

4. **Reduced Variance**: The model's predictions have lower variance than ground truth, suggesting it may not fully capture the diversity in quality scores.

5. **Distribution Mismatch**: The training set mean (0.62) differs significantly from the test set mean (0.46), which may contribute to performance issues.

---

## 6. Recommendations

### Immediate Actions

1. **Investigate Distribution Mismatch**: Analyze why the test set has a lower mean score than the training set. Consider:
   - Stratified sampling for train/test split
   - Ensuring test set distribution matches training distribution
   - Re-evaluating data augmentation strategies

2. **Address Low-Quality Prediction Bias**: 
   - Increase training samples in the low-quality range (0.2-0.4)
   - Apply class weighting or focal loss to emphasize low-quality samples
   - Consider separate models for different quality ranges

3. **Feature Engineering**:
   - Explore additional features beyond sentence embeddings
   - Consider hybrid features (structure + semantic)
   - Investigate domain-specific features for table extraction

### Model Improvements

1. **Architecture Enhancements**:
   - Experiment with deeper networks
   - Try attention mechanisms
   - Consider ensemble methods

2. **Training Strategy**:
   - Implement better regularization techniques
   - Use curriculum learning (train on easier samples first)
   - Apply data augmentation specifically for underrepresented ranges

3. **Loss Function**:
   - Experiment with Huber loss for robustness
   - Consider quantile regression for better uncertainty estimation
   - Try focal loss variants for imbalanced data

### Evaluation Improvements

1. **Additional Metrics**:
   - Calculate per-bucket accuracy
   - Analyze confusion matrix for quality ranges
   - Measure calibration curves

2. **Error Analysis**:
   - Identify common patterns in mispredictions
   - Analyze table structure features of problematic samples
   - Investigate domain-specific failure modes

---

## 7. Conclusion

The MLP-based table extraction quality prediction model shows **promising results** for practical applications where tolerance-based accuracy is acceptable. With **78.9% accuracy within ±10%** and **89.8% within ±15%**, the model can provide useful quality estimates for most table extraction scenarios.

However, the **negative R² score and moderate correlation** indicate significant room for improvement. The model's tendency to **overestimate low-quality extractions** is a critical limitation that should be addressed before deployment in production systems.

The enhanced dataset (Version 3) with improved distribution in the high-quality bucket (9,481 samples, 26.3%) has contributed to better training, but further improvements are needed to address the distribution mismatch between training and test sets and to improve performance on low-quality samples.

**Overall Assessment**: The model is **suitable for preliminary quality screening** but requires refinement before use in high-stakes applications where precise quality assessment is critical.

---

## Appendix

### Files Generated
- Model checkpoints: `experiments/mlp_regressor_v3/`
- Evaluation results: `results/test_evaluation.json`
- Visualization plots: `results/test_plots/`
- This report: `results/test_evaluation_report.md`

### Model Training Details
- Best model saved at epoch 84
- Training completed with early stopping at epoch 99
- Final validation MAE: 0.0528
- Final validation R²: 0.8648

---

**Report Generated**: December 2024  
**Model**: MLP Regressor v3  
**Dataset**: Version 3 (36,066 train, 3,000 test)

