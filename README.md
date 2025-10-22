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


# Prepare the JSON input from the dataset for eveluation model
python scripts/extract_tables_scitsr.py \
  --input data/SciTSR/test/img \
  --output data/SciTSR/test/json_output

# Start Jupyter Lab
./start_jupyter.sh
```

## 🙏 Acknowledgments

- **CS230 Teaching Team**: Stanford University
- **Hugging Face**: For the datasets library and infrastructure
