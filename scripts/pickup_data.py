#!/usr/bin/env python3
"""
Create a subset of PubTables-1M dataset by randomly selecting PMC documents.

This script:
1. Scans the full dataset to find unique PMC IDs
2. Randomly samples N PMC IDs
3. Copies all files (images + annotations) for those PMC IDs to a subset directory
"""

import os
import random
import shutil
from pathlib import Path
from collections import defaultdict


def get_unique_pmc_ids(source_dir):
    """
    Extract unique PMC IDs from the dataset.
    File format: PMC{number}_table_{n}.jpg/xml
    """
    print("🔍 Scanning for unique PMC IDs...")
    
    pmc_ids = set()
    images_dir = os.path.join(source_dir, "images")
    
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    
    # Scan all image files to get PMC IDs
    for filename in os.listdir(images_dir):
        if filename.startswith("PMC") and filename.endswith(".jpg"):
            # Extract PMC ID (e.g., "PMC1064074" from "PMC1064074_table_0.jpg")
            pmc_id = filename.split("_table_")[0]
            pmc_ids.add(pmc_id)
    
    print(f"✅ Found {len(pmc_ids)} unique PMC documents")
    return sorted(list(pmc_ids))


def get_files_for_pmc(source_dir, pmc_id):
    """
    Get all files (images and annotations) for a given PMC ID.
    Returns dict: {'images': [...], 'train': [...], 'test': [...], 'val': [...]}
    """
    files = defaultdict(list)
    
    # Pattern: PMC{id}_table_*.jpg or PMC{id}_table_*.xml
    pattern = f"{pmc_id}_table_"
    
    # Check each subdirectory
    for subdir in ['images', 'train', 'test', 'val']:
        dir_path = os.path.join(source_dir, subdir)
        if not os.path.exists(dir_path):
            continue
            
        for filename in os.listdir(dir_path):
            if filename.startswith(pattern):
                files[subdir].append(filename)
    
    return files


def create_subset(source_dir, output_dir, num_samples, seed=42):
    """
    Create a random subset of the dataset.
    
    Args:
        source_dir: Path to PubTables-1M-Structure directory
        output_dir: Path to output subset directory
        num_samples: Number of PMC documents to sample
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    print(f"\n📦 Creating subset with {num_samples} PMC documents")
    print(f"   Source: {source_dir}")
    print(f"   Output: {output_dir}")
    print()
    
    # Step 1: Get all unique PMC IDs
    all_pmc_ids = get_unique_pmc_ids(source_dir)
    
    if num_samples > len(all_pmc_ids):
        print(f"⚠️  Warning: Requested {num_samples} samples but only {len(all_pmc_ids)} available")
        num_samples = len(all_pmc_ids)
    
    # Step 2: Random sample
    print(f"🎲 Randomly sampling {num_samples} PMC IDs (seed={seed})...")
    selected_pmc_ids = random.sample(all_pmc_ids, num_samples)
    
    # Step 3: Create output directories
    print("📁 Creating output directories...")
    for subdir in ['images', 'train', 'test', 'val', 'words']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # Step 4: Save selected PMC IDs to file
    pmc_list_file = os.path.join(output_dir, "selected_pmc_ids.txt")
    with open(pmc_list_file, 'w') as f:
        for pmc_id in sorted(selected_pmc_ids):
            f.write(f"{pmc_id}\n")
    print(f"💾 Saved PMC ID list to: {pmc_list_file}")
    print()
    
    # Step 5: Copy files
    print("📋 Copying files...")
    stats = defaultdict(int)
    
    for i, pmc_id in enumerate(selected_pmc_ids, 1):
        if i % 100 == 0 or i == len(selected_pmc_ids):
            print(f"   Progress: {i}/{len(selected_pmc_ids)} PMC documents processed...")
        
        # Get all files for this PMC ID
        files = get_files_for_pmc(source_dir, pmc_id)
        
        # Copy files to output directories
        for subdir in ['images', 'train', 'test', 'val']:
            if subdir not in files:
                continue
                
            for filename in files[subdir]:
                src = os.path.join(source_dir, subdir, filename)
                dst = os.path.join(output_dir, subdir, filename)
                
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    stats[subdir] += 1
    
    print()
    print("=" * 60)
    print("✅ Subset creation complete!")
    print()
    print(f"📊 Statistics:")
    print(f"   PMC documents: {len(selected_pmc_ids)}")
    print(f"   Images copied: {stats['images']}")
    print(f"   Train annotations: {stats['train']}")
    print(f"   Test annotations: {stats['test']}")
    print(f"   Val annotations: {stats['val']}")
    print(f"   Total files: {sum(stats.values())}")
    print()
    print(f"📂 Output directory: {output_dir}")
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create a random subset of PubTables-1M dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a 1000-document subset
  python scripts/pickup_data.py --num 1000 --output data/pubtables_subset_1k
  
  # Create a 10000-document subset
  python scripts/pickup_data.py --num 10000 --output data/pubtables_subset_10k
  
  # Use custom seed for reproducibility
  python scripts/pickup_data.py --num 5000 --seed 123 --output data/pubtables_subset_5k
        """
    )
    
    parser.add_argument(
        "--source",
        default="data/pubtables_raw/PubTables-1M-Structure",
        help="Source directory with full dataset (default: data/pubtables_raw/PubTables-1M-Structure)"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for subset"
    )
    
    parser.add_argument(
        "--num",
        type=int,
        required=True,
        help="Number of PMC documents to sample"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate source directory
    if not os.path.exists(args.source):
        print(f"❌ Error: Source directory not found: {args.source}")
        print(f"   Make sure you've extracted the dataset first.")
        return 1
    
    # Create subset
    create_subset(args.source, args.output, args.num, args.seed)
    
    return 0


if __name__ == "__main__":
    exit(main())


#python script to pickup the data from the pubtables_raw directory and put it in the pubtables_subset directory.

"""
I would like to pick up the data from the data/pubtables_raw directory and put it in the data/pubtables_subset_1k directory.

The data/pubtables_subset_1k directory should have the following structure:

data/pubtables_subset_1k/
├── images/
├── train/
├── test/
├── val/
└── words/

The images directory should have the same structure as the pubtables_raw/images directory.

Randomly generate 1000 numbers from 1 to 4435253, make it a list, save the list to a file 
called pubtables_subset_numbers.txt in the pubtables_subset_1k directory.

1. copy the files in the images directory that starts with the PCMprefix + the numbers in the list + anything after the numbers 
from the pubtables_raw/images directory and put it in the pubtables_subset_1k/images directory.

2. copy the files in the train directory that starts with the PCMprefix + the numbers in the list + anything after the numbers 
from the pubtables_raw/train directory and put it in the pubtables_subset_1k/train directory.

3. copy the files in the test directory that starts with the PCMprefix + the numbers in the list + anything after the numbers 
from the pubtables_raw/test directory and put it in the pubtables_subset_1k/test directory.

4. copy the files in the val directory that starts with the PCMprefix + the numbers in the list + anything after the numbers 
from the pubtables_raw/val directory and put it in the pubtables_subset_1k/val directory.

Do another round, this time randomly generate 10000 numbers from 1 to 4435253, make it a list, make sure it is not 
duplicate with the previous list, save the list to a file called pubtables_subset_numbers_10k.txt in the pubtables_subset_10k directory.

Do the same for the train, test, and val directories.

If there is any error, print the error and exit the script.

"""