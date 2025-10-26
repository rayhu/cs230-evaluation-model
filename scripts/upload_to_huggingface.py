#!/usr/bin/env python3
"""
Upload the dataset to Hugging Face Hub.

Usage:
    python scripts/upload_to_huggingface.py --repo-id your-username/table-extraction-dataset
"""

import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, login

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


def upload_dataset(
    data_dir: Path,
    repo_id: str,
    private: bool = False,
    token: str | None = None,
) -> None:
    """
    Upload dataset to Hugging Face Hub.
    
    Args:
        data_dir: Directory containing Parquet files
        repo_id: Hugging Face repository ID (e.g., 'username/dataset-name')
        private: Whether the dataset should be private
        token: Hugging Face token (optional, will prompt if not provided)
    """
    # Load the dataset from Parquet files
    print(f"Loading dataset from {data_dir}...")
    dataset = load_dataset(str(data_dir))
    
    print(f"\nDataset loaded successfully!")
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Test samples: {len(dataset['test'])}")
    
    # Show an example
    print("\nExample from train split:")
    example = dataset['train'][0]
    print(f"  ID: {example['id']}")
    print(f"  Split: {example['split']}")
    print(f"  Similarity Score: {example['similarity_score']:.3f}")
    print(f"  Ground Truth Cells: {len(example['ground_truth']['cells'])}")
    print(f"  Generated Cells: {len(example['generated']['cells'])}")
    
    # Login to Hugging Face
    if token:
        login(token=token)
    else:
        print("\nPlease login to Hugging Face...")
        login()
    
    # Upload to hub
    print(f"\nUploading dataset to {repo_id}...")
    dataset.push_to_hub(
        repo_id=repo_id,
        private=private,
        commit_message="Initial dataset upload"
    )
    
    print(f"\n✅ Dataset uploaded successfully to: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (optional)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "dataset_parquet",
        help="Directory containing Parquet files"
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        print(f"Please run: python scripts/convert_to_parquet.py")
        return 1
    
    # Check for Parquet files
    parquet_files = list(args.data_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"Error: No Parquet files found in {args.data_dir}")
        print(f"Please run: python scripts/convert_to_parquet.py")
        return 1
    
    try:
        upload_dataset(
            data_dir=args.data_dir,
            repo_id=args.repo_id,
            private=args.private,
            token=args.token,
        )
        return 0
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

