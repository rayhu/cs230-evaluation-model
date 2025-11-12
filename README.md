# CS230 Table Evaluation Model

A deep learning project for evaluating table extraction performances.

## Overview

This project is a class project of Stanford's CS230 Deep Learning course, focusing on table understanding in scientific documents. We leverage the SciTSR dataset, which contains 15000 annotated table images from scientific papers.

## Project Structure

```
cs230-evaluation-model/
├── baseline/                       # Baseline model experiments and results
│   ├── mlp_hybrid_balanced/        # MLP with balanced regularization (dropout + weight decay)
├── notebooks/                      # Jupyter notebooks for exploration and training
│   ├── dataset_exploration.ipynb   # Dataset exploration and analysis
│   ├── eval_mlp_pytorch.ipynb      # PyTorch MLP model experiment
│   ├── eval_sentence_transformers.ipynb  # Sentence transformer experiments
│   ├── eval_transformers_hf.ipynb # Hugging Face transformer experiments
│   ├── table_quality_prediction_tensorflow_Base.ipynb  # TensorFlow model experiement
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
│   ├── structure_converter.py     # Table structure conversion utilities
│   └── utils/                     # Utility modules
│       ├── table_features.py     # Feature extraction utilities
│       └── grid_detection.py     # Grid detection algorithms
├── requirements.txt                # Python dependencies
├── setup.sh                        # Automated environment setup
├── start_jupyter.sh                # Jupyter Lab launcher
├── SETUP.md                        # Detailed setup guide
└── README.md                       # This file
```

## Prerequisites

- Python 3.13 specified in venv

## Setup

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


# Start Jupyter Lab
./start_jupyter.sh
```

## Project Goal: Neural Verifier

**Objective**: Build a neural network that can predict table extraction quality **without** ground truth.

## Baseline Model

The `baseline/` directory contains trained baseline model and experimental result. It contains:

- **Configuration files** (`config.json`): Hyperparameters and training settings
- **Training history** (`training_history.json`, `training_history.png`): Loss curves and metrics over training epochs
- **Loss plots** (`loss_vs_epoch.png`): Visualization of training and validation loss
- **Test evaluations** (where available): Final model performance on test set

## Dataset Available on Hugging Face

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

## Acknowledgments

- **SciTSR Dataset**: [Academic-Hammer/SciTSR](https://github.com/Academic-Hammer/SciTSR)
- **Table Transformer**: [microsoft/table-transformer](https://huggingface.co/microsoft/table-transformer-structure-recognition)
- **EasyOCR**: [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)
- **CS230 Teaching Team**: Stanford University
- **Hugging Face**: For the datasets library and infrastructure
