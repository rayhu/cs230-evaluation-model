#!/usr/bin/env python3
"""
Convert dataset to Parquet format for Hugging Face Hub upload.

This script converts the JSON-based dataset into Parquet format that
can be easily loaded with datasets.load_dataset().
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def load_json_data(file_path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_split(
    metadata_file: Path,
    generated_dir: Path,
    gt_dir: Path,
    split_name: str,
) -> list[Dict[str, Any]]:
    """
    Process a dataset split and return list of examples.
    
    Args:
        metadata_file: Path to metadata JSONL file
        generated_dir: Directory containing generated JSON files
        gt_dir: Directory containing ground truth JSON files
        split_name: Name of the split (train/test)
    
    Returns:
        List of example dictionaries
    """
    examples = []
    
    print(f"\nProcessing {split_name} split...")
    print(f"  Reading metadata from: {metadata_file}")
    
    # Read metadata
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"  Processing {split_name}"):
            try:
                metadata = json.loads(line.strip())
                
                # Load generated JSON
                generated_file = generated_dir / metadata["generated_file"]
                if not generated_file.exists():
                    print(f"Warning: Generated file not found: {generated_file}")
                    continue
                    
                generated_data = load_json_data(generated_file)
                
                # Load ground truth JSON
                gt_file = gt_dir / metadata["ground_truth_file"]
                if not gt_file.exists():
                    print(f"Warning: Ground truth file not found: {gt_file}")
                    continue
                    
                ground_truth_data = load_json_data(gt_file)
                
                # Create example
                example = {
                    "id": metadata["id"],
                    "split": split_name,
                    "similarity_score": metadata["similarity_score"],
                    "ground_truth": ground_truth_data,
                    "generated": generated_data,
                }
                
                examples.append(example)
                
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
    
    return examples


def convert_to_parquet(
    dataset_dir: Path,
    output_dir: Path,
) -> None:
    """
    Convert dataset to Parquet format.
    
    Args:
        dataset_dir: Directory containing the dataset
        output_dir: Directory to save Parquet files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process train split
    train_dir = dataset_dir / "train"
    train_examples = process_split(
        metadata_file=train_dir / "metadata_train.jsonl",
        generated_dir=train_dir / "generated",
        gt_dir=train_dir / "ground_truth",
        split_name="train",
    )
    
    # Process test split
    test_dir = dataset_dir / "test"
    test_examples = process_split(
        metadata_file=test_dir / "metadata_test.jsonl",
        generated_dir=test_dir / "generated",
        gt_dir=test_dir / "ground_truth",
        split_name="test",
    )
    
    # Convert to Arrow tables
    print("\nConverting to Arrow/Parquet format...")
    
    # Convert train split
    print(f"Converting train split ({len(train_examples)} examples)...")
    train_table = pa.Table.from_pylist(train_examples)
    train_path = output_dir / "train-00000-of-00001.parquet"
    pq.write_table(train_table, train_path)
    print(f"  Saved: {train_path}")
    
    # Convert test split
    print(f"Converting test split ({len(test_examples)} examples)...")
    test_table = pa.Table.from_pylist(test_examples)
    test_path = output_dir / "test-00000-of-00001.parquet"
    pq.write_table(test_table, test_path)
    print(f"  Saved: {test_path}")
    
    # Create dataset card
    dataset_card = output_dir / "README.md"
    with open(dataset_card, "w") as f:
        f.write(f"""# Table Extraction Evaluation Dataset

## Dataset Summary

This dataset contains {len(train_examples) + len(test_examples)} examples of table extraction evaluation data.

- **Train split**: {len(train_examples)} examples
- **Test split**: {len(test_examples)} examples

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
@software{{table_extraction_evaluation_2025,
  author = {{Ray Hu, Hiva Zaad, Nofel Teldjoune}},
  title = {{Table Extraction Evaluation Dataset}},
  year = {{2025}},
  url = {{https://github.com/rayhu/cs230-evaluation-model}}
}}
```
""")
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETE")
    print("="*70)
    print(f"Train examples: {len(train_examples)}")
    print(f"Test examples: {len(test_examples)}")
    print(f"\nParquet files saved to: {output_dir}")
    print(f"\nTo upload to Hugging Face Hub:")
    print(f"  python scripts/upload_to_huggingface.py --repo-id your-username/dataset-name --parquet-dir {output_dir}")
    print("\nOr load locally:")
    print(f"  dataset = load_dataset('path/to/{output_dir.name}')")


def main():
    parser = argparse.ArgumentParser(
        description="Convert dataset to Parquet format for Hugging Face"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).parent.parent / "dataset",
        help="Dataset directory containing train/test splits"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "dataset_parquet",
        help="Output directory for Parquet files"
    )
    
    args = parser.parse_args()
    
    # Validate dataset directory
    if not args.dataset_dir.exists():
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        return 1
    
    # Convert to Parquet
    try:
        convert_to_parquet(args.dataset_dir, args.output_dir)
        return 0
    except Exception as e:
        print(f"Error converting dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

