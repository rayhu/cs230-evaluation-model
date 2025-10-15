# CS230 Table Evaluation Model

A deep learning project for evaluating table extraction performances.

## 🎯 Overview

This project is part of Stanford's CS230 Deep Learning course, focusing on table understanding in scientific documents. We leverage the PubTables-1M dataset, which contains 1 million annotated table images from scientific papers.

## 📁 Project Structure

```
cs230-evaluation-model/
├── data/                           # Dataset storage (gitignored)
│   ├── README.md                   # Dataset documentation
│   └── pubtables_raw/              # Downloaded data
├── docs/                           # Project documentation
├── experiments/                    # Experiment configs and results
├── notebooks/                      # Jupyter notebooks for exploration
│   └── 01_dataset_exploration.ipynb
├── scripts/                        # Utility scripts
|   ├── extract_structure_dataset.sh
│   └── download_pubtables_raw.py
├── src/                            # Source code
│   └── main.py
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

# Run automated setup
./setup.sh

# Download the dataset and extract them
python scripts/download_pubtables_raw.py --output data/pubtables_raw
bash scripts/extract_structure_dataset.sh

# Start Jupyter Lab
./start_jupyter.sh
```

## 🙏 Acknowledgments

- **PubTables-1M Dataset**: [bsmock/pubtables-1m](https://huggingface.co/datasets/bsmock/pubtables-1m)
- **CS230 Teaching Team**: Stanford University
- **Hugging Face**: For the datasets library and infrastructure
