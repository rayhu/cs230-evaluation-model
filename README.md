# CS230 Table Evaluation Model

A deep learning project for evaluating table extraction performances.

## 🎯 Overview

This project is part of Stanford's CS230 Deep Learning course, focusing on table understanding in scientific documents. We leverage the SciTSR dataset, which contains 15000 annotated table images from scientific papers.

## 📁 Project Structure

```
cs230-evaluation-model/
├── data/                           # Dataset storage (gitignored)
│   ├── README.md                   # Dataset documentation
│   └── SciTSR/                     # Downloaded data
├── docs/                           # Project documentation
├── experiments/                    # Experiment configs and results
├── notebooks/                      # Jupyter notebooks for exploration
│   └── 01_dataset_exploration.ipynb
├── scripts/                        # Utility scripts
│   └── extract_tables_scitsr.py    # Prepare the input JSON files for evalution model
├── requirements.txt                # Python dependencies
├── setup.sh                        # Automated environment setup
├── start_jupyter.sh                # Jupyter Lab launcher
├── SETUP.md                        # Detailed setup guide
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.13 specified in venv
- 120 GB free disk space

### Setup

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd cs230-evaluation-model

# Run automated setup or manually install using uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

or

./setup.sh

# Download the SciTSR dataset and extract them to data folder

# Start Jupyter Lab
./start_jupyter.sh
```

## 📊 Table Extraction & Evaluation Workflow

### 1. Extract Tables from Images

Process SciTSR test images using Table Transformer + EasyOCR (GPU-accelerated):

```bash
# Test on single image
python scripts/extract_tables_scitsr.py \
  --single data/SciTSR/test/img/0704.1068v2.1.png \
  --output data/SciTSR/test/json_output

# Process all 3000 test images (~8-10 hours on Apple Silicon MPS)
python scripts/extract_tables_scitsr.py \
  --input data/SciTSR/test/img \
  --output data/SciTSR/test/json_output

# Test with first 10 images
python scripts/extract_tables_scitsr.py \
  --input data/SciTSR/test/img \
  --output data/SciTSR/test/json_output \
  --limit 10
```

### 2. Evaluate Extraction Quality

Compare extracted tables with ground truth using multiple metrics:

```bash
# Evaluate single file
python scripts/score_extraction.py \
  --pred data/SciTSR/test/json_output/0704.1068v2.1.json \
  --gt data/SciTSR/test/structure/0704.1068v2.1.json \
  --detailed

# Batch evaluation (all files)
python scripts/score_extraction.py \
  --pred data/SciTSR/test/json_output \
  --gt data/SciTSR/test/structure \
  --output results/evaluation_scores.json
```

**Evaluation Metrics Provided:**
- **Cell Detection**: Precision, Recall, F1 (IoU-based matching)
- **Content Accuracy**: Text similarity, exact match rate  
- **Structure Accuracy**: Row/column detection accuracy
- **Overall Score**: Weighted combination (0-1 scale)

📖 See [`docs/EVALUATION_GUIDE.md`](docs/EVALUATION_GUIDE.md) for detailed explanation of metrics.

### 3. Validate Output Format

Check JSON format validity and statistics:

```bash
python scripts/validate_outputs.py \
  --output-dir data/SciTSR/test/json_output \
  --gt-dir data/SciTSR/test/structure \
  --save-report results/validation_report.json
```

## 🎯 Project Goal: Neural Verifier

**Objective**: Build a neural network that can predict table extraction quality **without** ground truth.

**Pipeline:**
1. ✅ Extract tables from 3000 test images → Generate predictions
2. ✅ Score predictions against ground truth → Get quality metrics (0-1 scores)
3. ⏭️ Train neural verifier: (table_image, extracted_json) → predicted_quality_score
4. ⏭️ Deploy: Automatically assess new extractions without manual annotation

**Your contribution**: The scoring system and extracted data will be training labels for the verifier model.

## 📚 Documentation

- [`docs/EVALUATION_GUIDE.md`](docs/EVALUATION_GUIDE.md) - Complete evaluation metrics guide
- [`docs/proposal/`](docs/proposal/) - Project proposal PDF
- [`SETUP.md`](SETUP.md) - Detailed setup instructions

## 🙏 Acknowledgments

- **SciTSR Dataset**: [Academic-Hammer/SciTSR](https://github.com/Academic-Hammer/SciTSR)
- **Table Transformer**: [microsoft/table-transformer](https://huggingface.co/microsoft/table-transformer-structure-recognition)
- **EasyOCR**: [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)
- **CS230 Teaching Team**: Stanford University
- **Hugging Face**: For the datasets library and infrastructure
