# Quick Start Guide: Table Extraction Evaluation

## 🎯 What You Have Now

A complete pipeline to:
1. **Extract** table structures from images (Table Transformer + EasyOCR)
2. **Score** extraction quality against ground truth (multi-metric evaluation)
3. **Validate** output formats and generate reports

This provides the foundation for training your **Neural Verifier** model.

## ⚡ Quick Commands

### Test on One Image (10 seconds)

```bash
# Extract table
python scripts/extract_tables_scitsr.py \
  --single data/SciTSR/test/img/0704.1068v2.1.png \
  --output data/SciTSR/test/json_output

# Score quality
python scripts/score_extraction.py \
  --pred data/SciTSR/test/json_output/0704.1068v2.1.json \
  --gt data/SciTSR/test/structure/0704.1068v2.1.json \
  --detailed
```

**Example Output:**
```
📊 OVERALL SCORE: 0.225 (22.5%)

🔍 Cell Detection: F1=0.0% (cells merged/wrong boundaries)
📝 Content Accuracy: 0% (no matched cells for text comparison)
🏗️  Structure Accuracy: 75% (row/col counts close but not exact)
```

### Process 10 Images for Testing (~2 minutes)

```bash
python scripts/extract_tables_scitsr.py \
  --input data/SciTSR/test/img \
  --output data/SciTSR/test/json_output \
  --limit 10
```

### Full Batch: All 3000 Images (~8-10 hours)

```bash
# Run extraction (can run overnight)
python scripts/extract_tables_scitsr.py \
  --input data/SciTSR/test/img \
  --output data/SciTSR/test/json_output

# Generate all evaluation scores
python scripts/score_extraction.py \
  --pred data/SciTSR/test/json_output \
  --gt data/SciTSR/test/structure \
  --output results/all_scores.json
```

## 📊 Understanding the Scores

### Overall Score Components (0-1 scale)

```
Overall Score = 
  40% × Cell Detection F1    (Are cells correctly identified?)
  30% × Text Similarity      (Is the content accurate?)
  15% × Row Accuracy         (Right number of rows?)
  15% × Column Accuracy      (Right number of columns?)
```

### Score Interpretation

| Score | Quality | Meaning |
|-------|---------|---------|
| 0.8-1.0 | ✅ Excellent | Near-perfect extraction |
| 0.6-0.8 | 👍 Good | Minor errors, mostly usable |
| 0.4-0.6 | 😐 Fair | Significant issues, needs correction |
| 0.2-0.4 | ⚠️ Poor | Major structural problems |
| 0.0-0.2 | ❌ Failed | Unusable extraction |

### Example: Your Test Case

**File:** `0704.1068v2.1.json`
- **Score:** 0.225 (Poor)
- **Why?** Detected 7 cells vs 12 in ground truth, cells are merged
- **Expected?** Yes! This is training data - errors help the model learn

## 🔄 Your Next Steps

### Option 1: Start Small (Recommended)

```bash
# 1. Extract 100 test images (~15 minutes)
python scripts/extract_tables_scitsr.py \
  --input data/SciTSR/test/img \
  --output data/SciTSR/test/json_output \
  --limit 100

# 2. Score them all (~1 minute)
python scripts/score_extraction.py \
  --pred data/SciTSR/test/json_output \
  --gt data/SciTSR/test/structure \
  --output results/scores_100.json

# 3. Analyze the distribution
python -c "
import json
with open('results/scores_100.json') as f:
    data = json.load(f)
    avg = data['averages']['overall_score']
    print(f'Average Score: {avg:.3f}')
    print('Score Distribution:')
    for bucket, count in data['distribution'].items():
        print(f'  {bucket}: {count}')
"
```

### Option 2: Full Dataset (8-10 hours)

```bash
# Run extraction overnight
nohup python scripts/extract_tables_scitsr.py \
  --input data/SciTSR/test/img \
  --output data/SciTSR/test/json_output \
  > extraction.log 2>&1 &

# Check progress
tail -f extraction.log

# After completion, score everything
python scripts/score_extraction.py \
  --pred data/SciTSR/test/json_output \
  --gt data/SciTSR/test/structure \
  --output results/all_scores.json
```

## 🧠 Building Your Neural Verifier

Once you have scored data, you can train a model to predict quality:

```python
# Pseudo-code for your next phase
import torch
from transformers import ViTModel  # Vision model for table images

class TableQualityVerifier(nn.Module):
    def __init__(self):
        # Vision encoder for table image
        self.vision_encoder = ViTModel.from_pretrained('...')
        # Structure encoder for extracted JSON
        self.structure_encoder = TransformerEncoder(...)
        # Quality predictor
        self.quality_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, image, extracted_structure):
        # Extract image features
        img_features = self.vision_encoder(image)
        # Extract structure features  
        struct_features = self.structure_encoder(extracted_structure)
        # Combine and predict quality score
        combined = torch.cat([img_features, struct_features])
        quality_score = self.quality_head(combined)
        return quality_score

# Training
for image, extracted_json, gt_json in dataset:
    # Get ground truth quality score from our evaluation
    gt_score = evaluate_extraction(extracted_json, gt_json)
    
    # Predict quality
    pred_score = model(image, extracted_json)
    
    # Train to match ground truth scores
    loss = mse_loss(pred_score, gt_score)
    loss.backward()
```

## 📁 Generated Files Structure

```
data/SciTSR/test/
├── img/                    # Original PNG images (3000 files)
├── structure/              # Ground truth JSON (3000 files)
└── json_output/            # Your extractions (3000 files)
    ├── 0704.1068v2.1.json
    ├── 0704.3573v1.3.json
    └── ...

results/
├── all_scores.json         # Complete evaluation results
├── validation_report.json  # Format validation stats
└── extraction_stats.json   # Processing statistics
```

## 🔧 Troubleshooting

### Extraction is slow
- **On CPU**: ~15-20 seconds per image (normal)
- **On MPS**: ~8-10 seconds per image (Apple Silicon GPU)
- **On CUDA**: ~2-3 seconds per image (NVIDIA GPU)

### Low scores are normal
- **Expected**: 0.3-0.6 average for Table Transformer on SciTSR
- **Why**: Scientific tables are complex, detection errors common
- **That's OK**: Your verifier will learn from these errors!

### Memory issues
```bash
# Process in smaller batches
for i in {0..29}; do
    python scripts/extract_tables_scitsr.py \
      --input data/SciTSR/test/img \
      --output data/SciTSR/test/json_output \
      --limit 100 \
      --resume
done
```

## 📖 Further Reading

- [`EVALUATION_GUIDE.md`](EVALUATION_GUIDE.md) - Detailed metrics explanation
- [Table Transformer Paper](https://arxiv.org/abs/2110.00061)
- [SciTSR Dataset Paper](https://arxiv.org/abs/1902.10031)

## 💡 Tips

1. **Start small**: Test with 10-100 images before full batch
2. **Check GPU**: Use `--device auto` to auto-detect best device
3. **Resume capability**: Script skips already-processed files
4. **Save intermediate**: Use `--output` to save scores regularly
5. **Adjust threshold**: Try `--iou-threshold 0.3` for lenient matching

---

**Ready to go?** Run the test extraction command above and see your first score! 🚀

