# CS230 Table Evaluation Model

A deep learning project for evaluating table extraction performances.

## 🎯 Overview

This project is a class project of Stanford's CS230 Deep Learning course, focusing on table understanding in scientific documents. We leverage the SciTSR dataset, which contains 15000 annotated table images from scientific papers.

## 📁 Project Structure

```
cs230-evaluation-model/
├── notebooks/                      # Jupyter notebooks for exploration and training
│   ├── dataset_exploration.ipynb   # Dataset exploration and analysis
│   ├── eval_mlp_pytorch.ipynb     # PyTorch MLP model evaluation
│   ├── eval_sentence_transformers.ipynb  # Sentence transformer experiments
│   ├── eval_transformers_hf.ipynb # Hugging Face transformer experiments
│   ├── table_quality_prediction_tensorflow_Base.ipynb  # TensorFlow baseline model
│   ├── table_quality_prediction_tensorflow_WEIGHTED.ipynb  # TensorFlow weighted loss model
│   └── experiments/                # Experiment results and checkpoints
├── scripts/                        # Utility scripts
│   ├── extract_tables_scitsr.py   # Extract tables from SciTSR images
│   ├── score_extraction.py        # Evaluate extraction quality scores
│   ├── score_extraction_improved.py  # Improved scoring implementation
│   ├── train_mlp_regressor.py     # Train MLP regressor model
│   ├── evaluate_on_test_set.py   # Evaluate model on test set
│   ├── predict_quality.py          # Predict quality for new tables
│   ├── validate_outputs.py        # Validate JSON output format
│   ├── generate_metadata_jsonl.py # Generate metadata for dataset
│   ├── convert_to_parquet.py      # Convert dataset to parquet format
│   ├── hyperparam_search.py       # Hyperparameter search utilities
│   ├── compare_models.py          # Compare different model architectures
│   ├── upload_to_huggingface.py   # Upload dataset to Hugging Face
│   └── train_*.sh                  # Training shell scripts
├── src/                            # Source code for models and utilities
│   ├── mlp_regressor.py           # MLP regressor model implementations
│   ├── structure_converter.py     # Table structure conversion utilities
│   ├── main.py                    # Main entry point
│   └── utils/                     # Utility modules
│       ├── table_features.py     # Feature extraction utilities
│       └── grid_detection.py     # Grid detection algorithms
├── requirements.txt                # Python dependencies
├── setup.sh                        # Automated environment setup
├── start_jupyter.sh                # Jupyter Lab launcher
├── SETUP.md                        # Detailed setup guide
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.13 specified in venv

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


# Prepare the JSON input from the dataset for eveluation model
python scripts/extract_tables_scitsr.py \
  --input data/SciTSR/train/img \
  --output data/SciTSR/train/json_output

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
3. ✅ Train neural verifier: (table_image, extracted_json) → predicted_quality_score
4. ✅ Deploy: Automatically assess new extractions without manual annotation

**Your contribution**: The scoring system and extracted data will be training labels for the verifier model.

### 🤖 MLP Regressor (Simple Baseline Model)

We've implemented a simple TF-IDF + MLP baseline model that predicts quality scores from table JSON alone:

```bash
# Train the model
python scripts/train_mlp_regressor.py \
  --epochs 10 \
  --output-dir experiments/mlp_regressor

# Test on the test set
python scripts/evaluate_on_test_set.py   --model-dir experiments/mlp_regressor   --output custom_results/evaluation.json   --plot-dir custom_results/plots
```

**Architecture**: JSON → TF-IDF (10k features) → MLP (256→64→1) → Quality Score

📖 See [`docs/MLP_REGRESSOR_GUIDE.md`](docs/MLP_REGRESSOR_GUIDE.md) for complete training and usage guide.

## 📦 Dataset Available on Hugging Face

Our table extraction evaluation dataset is now available on Hugging Face Hub!

**Dataset**: [rayhu/table-extraction-evaluation](https://huggingface.co/datasets/rayhu/table-extraction-evaluation)

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("rayhu/table-extraction-evaluation")

# Access splits
train = dataset['train']  # 11,971 examples
test = dataset['test']    # 3,000 examples
```

📖 See [`DATASET_USAGE.md`](DATASET_USAGE.md) for detailed usage instructions.

## 📚 Documentation

- [`DATASET_USAGE.md`](DATASET_USAGE.md) - How to use the dataset
- [`docs/EVALUATION_GUIDE.md`](docs/EVALUATION_GUIDE.md) - Complete evaluation metrics guide
- [`docs/MLP_REGRESSOR_GUIDE.md`](docs/MLP_REGRESSOR_GUIDE.md) - MLP baseline model training and usage
- [`docs/proposal/`](docs/proposal/) - Project proposal PDF
- [`SETUP.md`](SETUP.md) - Detailed setup instructions

## 🙏 Acknowledgments

- **SciTSR Dataset**: [Academic-Hammer/SciTSR](https://github.com/Academic-Hammer/SciTSR)
- **Table Transformer**: [microsoft/table-transformer](https://huggingface.co/microsoft/table-transformer-structure-recognition)
- **EasyOCR**: [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)
- **CS230 Teaching Team**: Stanford University
- **Hugging Face**: For the datasets library and infrastructure
