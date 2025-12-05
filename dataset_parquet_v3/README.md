# Table Extraction Evaluation Dataset

## Dataset Summary

This dataset contains 39066 examples of table extraction evaluation data.

- **Train split**: 36066 examples
- **Test split**: 3000 examples

## Dataset Structure

Each example contains:
- `id`: Unique identifier for the sample
- `split`: Either "train" or "test"
- `similarity_score`: Quality score between 0 and 1
- `ground_truth`: Ground truth table structure
- `generated`: Generated table structure

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("path/to/parquet/files", data_dir=".")
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
