#!/usr/bin/env python3
"""
Upload dataset to Hugging Face Hub using a write token.

Usage:
    # Set your token first
    export HF_TOKEN="your_write_token_here"
    
    # Or pass it directly
    HF_TOKEN="your_write_token_here" python scripts/upload_with_token.py
"""

import os
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import login

# Get token from environment
token = os.environ.get("HF_TOKEN", "YOUR_WRITE_TOKEN_HERE")

if token == "YOUR_WRITE_TOKEN_HERE":
    print("❌ Error: No token provided!")
    print("\nPlease set your write token:")
    print("  export HF_TOKEN='your_write_token_here'")
    print("\nOr edit this script and replace YOUR_WRITE_TOKEN_HERE with your token.")
    sys.exit(1)

# Login
print("Logging in to Hugging Face...")
login(token=token)

# Load dataset
data_dir = Path(__file__).parent.parent / "dataset_parquet"
print(f"\nLoading dataset from {data_dir}...")
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

# Upload to hub
repo_id = "rayhu/table-extraction-evaluation"
print(f"\nUploading dataset to {repo_id}...")

try:
    dataset.push_to_hub(
        repo_id=repo_id,
        private=False,
        commit_message="Initial dataset upload: 11,971 train + 3,000 test samples"
    )
    
    print(f"\n✅ Dataset uploaded successfully!")
    print(f"\n   View at: https://huggingface.co/datasets/{repo_id}")
    print(f"\n   Load it with:")
    print(f"   dataset = load_dataset('{repo_id}')")
    
except Exception as e:
    print(f"\n❌ Error uploading dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

