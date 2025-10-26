# Table Extraction Evaluation Dataset - Usage Guide

**Dataset on Hugging Face**: [rayhu/table-extraction-evaluation](https://huggingface.co/datasets/rayhu/table-extraction-evaluation)

## Quick Start

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("rayhu/table-extraction-evaluation")

# Access splits
train = dataset['train']  # 11,971 examples
test = dataset['test']     # 3,000 examples
```

## Dataset Structure

Each example contains:
- `id`: Unique identifier
- `split`: "train" or "test"
- `similarity_score`: Quality score (0-1) based on IoU matching
- `ground_truth`: Ground truth table structure with cells
- `generated`: Generated/extracted table structure with cells

### Example Usage

```python
from datasets import load_dataset

dataset = load_dataset("rayhu/table-extraction-evaluation")

# Get an example
example = dataset['train'][0]

# Access fields
print(f"ID: {example['id']}")
print(f"Similarity Score: {example['similarity_score']:.3f}")

# Access table cells
ground_truth = example['ground_truth']['cells']
generated = example['generated']['cells']

# Each cell has: id, tex, content, start_row, end_row, start_col, end_col
```

## Use Cases

### 1. Training a Neural Verifier

```python
dataset = load_dataset("rayhu/table-extraction-evaluation")

for example in dataset['train']:
    # Input: (generated table structure)
    # Target: similarity_score
    
    input_data = example['generated']
    target_score = example['similarity_score']
    
    # Train your model to predict similarity_score
```

### 2. Analyzing Extraction Quality

```python
import numpy as np

dataset = load_dataset("rayhu/table-extraction-evaluation")
scores = [ex['similarity_score'] for ex in dataset['train']]

print(f"Mean score: {np.mean(scores):.3f}")
print(f"Max score: {np.max(scores):.3f}")
print(f"Min score: {np.min(scores):.3f}")
```

### 3. Comparing Extractions

```python
dataset = load_dataset("rayhu/table-extraction-evaluation")

for example in dataset['test']:
    gt_cells = example['ground_truth']['cells']
    gen_cells = example['generated']['cells']
    
    # Compare ground truth vs generated
    print(f"Cells: GT={len(gt_cells)}, Generated={len(gen_cells)}")
```

## Citation

```bibtex
@software{table_extraction_evaluation_2025,
  author = {Ray Hu, Hiva Zaad, Nofel Teldjoune},
  title = {Table Extraction Evaluation Dataset},
  year = {2025},
  url = {https://github.com/rayhu/cs230-evaluation-model}
}
```

## Repository

- **Hugging Face**: https://huggingface.co/datasets/rayhu/table-extraction-evaluation
- **GitHub**: https://github.com/rayhu/cs230-evaluation-model
- **License**: Apache 2.0

