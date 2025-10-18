#!/usr/bin/env python3
"""
Create a subset of PubTables-1M dataset by randomly selecting PMC documents.

This script reads from filelist.txt files for fast processing and validates
that all files exist before copying.
"""

import os
import random
import shutil
from pathlib import Path
from collections import defaultdict


def load_filelists(source_dir):
    """
    Load all filelists from the dataset.
    Returns: dict of {split: set of file paths}
    """
    print("📖 Loading filelists...")
    
    filelists = {
        'images': set(),
        'train': set(),
        'test': set(),
        'val': set()
    }
    
    # Load each filelist
    filelist_files = {
        'images': 'images_filelist.txt',
        'train': 'train_filelist.txt',
        'test': 'test_filelist.txt',
        'val': 'val_filelist.txt'
    }
    
    for split, filename in filelist_files.items():
        filepath = os.path.join(source_dir, filename)
        if not os.path.exists(filepath):
            print(f"   ⚠️  Warning: {filename} not found, skipping...")
            continue
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    filelists[split].add(line)
        
        print(f"   ✅ Loaded {len(filelists[split])} files from {filename}")
    
    return filelists


def extract_pmc_ids_from_filelists(filelists):
    """
    Extract unique PMC IDs from the image filelist.
    Returns: sorted list of PMC IDs
    """
    print("\n🔍 Extracting unique PMC IDs...")
    
    pmc_ids = set()
    
    for filepath in filelists['images']:
        # Format: images/PMC1064074_table_0.jpg
        filename = os.path.basename(filepath)
        if filename.startswith('PMC') and '_table_' in filename:
            # Extract PMC ID
            pmc_id = filename.split('_table_')[0]
            pmc_ids.add(pmc_id)
    
    print(f"✅ Found {len(pmc_ids)} unique PMC documents")
    return sorted(list(pmc_ids))


def get_files_for_pmc_from_filelists(filelists, pmc_id):
    """
    Get all files for a given PMC ID from the filelists.
    Returns: dict mapping base_name -> {'image': path, 'annotation': (split, path)}
    """
    matches = {}
    
    # Pattern: PMC{id}_table_
    pattern = f"{pmc_id}_table_"
    
    # Find all image files for this PMC
    image_files = {}
    for filepath in filelists['images']:
        filename = os.path.basename(filepath)
        if filename.startswith(pattern):
            base_name = filename.rsplit('.', 1)[0]  # Remove .jpg
            image_files[base_name] = filepath
    
    # Find all annotation files for this PMC
    annotation_files = {}
    for split in ['train', 'test', 'val']:
        for filepath in filelists[split]:
            filename = os.path.basename(filepath)
            if filename.startswith(pattern):
                base_name = filename.rsplit('.', 1)[0]  # Remove .xml
                if base_name in annotation_files:
                    # Duplicate annotation - this shouldn't happen
                    pass  # We'll handle in validation
                else:
                    annotation_files[base_name] = (split, filepath)
    
    # Match images with annotations
    for base_name, image_path in image_files.items():
        if base_name in annotation_files:
            split, ann_path = annotation_files[base_name]
            matches[base_name] = {
                'image': image_path,
                'annotation': (split, ann_path)
            }
    
    return matches


def create_subset(source_dir, output_dir, num_samples, seed=42, verify_files=True):
    """
    Create a random subset of the dataset using filelists.
    
    Args:
        source_dir: Path to PubTables-1M-Structure directory
        output_dir: Path to output subset directory
        num_samples: Number of PMC documents to sample
        seed: Random seed for reproducibility
        verify_files: If True, verify files exist before copying
    """
    random.seed(seed)
    
    print(f"\n📦 Creating subset with {num_samples} PMC documents")
    print(f"   Source: {source_dir}")
    print(f"   Output: {output_dir}")
    print()
    
    # Step 1: Load filelists
    filelists = load_filelists(source_dir)
    
    # Step 2: Extract unique PMC IDs
    all_pmc_ids = extract_pmc_ids_from_filelists(filelists)
    
    if num_samples > len(all_pmc_ids):
        print(f"⚠️  Warning: Requested {num_samples} samples but only {len(all_pmc_ids)} available")
        num_samples = len(all_pmc_ids)
    
    # Step 3: Random sample
    print(f"\n🎲 Randomly sampling {num_samples} PMC IDs (seed={seed})...")
    selected_pmc_ids = random.sample(all_pmc_ids, num_samples)
    
    # Step 4: Create output directories
    print("📁 Creating output directories...")
    for subdir in ['images', 'train', 'test', 'val', 'words']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # Step 5: Save selected PMC IDs to file
    pmc_list_file = os.path.join(output_dir, "selected_pmc_ids.txt")
    with open(pmc_list_file, 'w') as f:
        for pmc_id in sorted(selected_pmc_ids):
            f.write(f"{pmc_id}\n")
    print(f"💾 Saved PMC ID list to: {pmc_list_file}")
    
    # Step 6: Collect all files to copy
    print(f"\n📋 Collecting files for {num_samples} PMC documents...")
    files_to_copy = []  # List of (src, dst) tuples
    stats = defaultdict(int)
    validation_errors = []
    
    for i, pmc_id in enumerate(selected_pmc_ids, 1):
        if i % 100 == 0 or i == len(selected_pmc_ids):
            print(f"   Progress: {i}/{len(selected_pmc_ids)} PMC documents processed...")
        
        # Get matched files from filelists
        matches = get_files_for_pmc_from_filelists(filelists, pmc_id)
        
        if not matches:
            validation_errors.append(f"{pmc_id}: No matching files found")
            continue
        
        for base_name, match_info in matches.items():
            # Prepare image copy
            image_relpath = match_info['image']
            image_src = os.path.join(source_dir, image_relpath)
            image_dst = os.path.join(output_dir, image_relpath)
            
            # Prepare annotation copy
            split, ann_relpath = match_info['annotation']
            ann_src = os.path.join(source_dir, ann_relpath)
            ann_dst = os.path.join(output_dir, ann_relpath)
            
            # Verify files exist if requested
            if verify_files:
                if not os.path.exists(image_src):
                    validation_errors.append(f"{pmc_id}: Image file not found: {image_relpath}")
                    continue
                if not os.path.exists(ann_src):
                    validation_errors.append(f"{pmc_id}: Annotation file not found: {ann_relpath}")
                    continue
            
            # Add to copy list
            files_to_copy.append((image_src, image_dst, 'images'))
            files_to_copy.append((ann_src, ann_dst, split))
            stats[f'{split}_matched'] += 1
    
    print(f"\n📊 Files to copy: {len(files_to_copy)}")
    
    # Step 7: Copy files
    print("📋 Copying files...")
    copied_stats = defaultdict(int)
    
    for i, (src, dst, file_type) in enumerate(files_to_copy, 1):
        if i % 500 == 0 or i == len(files_to_copy):
            print(f"   Progress: {i}/{len(files_to_copy)} files copied...")
        
        try:
            shutil.copy2(src, dst)
            copied_stats[file_type] += 1
        except Exception as e:
            validation_errors.append(f"Copy error: {src} -> {dst}: {e}")
    
    # Step 8: Save validation errors if any
    if validation_errors:
        error_file = os.path.join(output_dir, "validation_errors.txt")
        with open(error_file, 'w') as f:
            for error in validation_errors:
                f.write(f"{error}\n")
        print(f"\n⚠️  {len(validation_errors)} validation errors found. See: {error_file}")
    
    # Step 9: Print summary
    print()
    print("=" * 60)
    print("✅ Subset creation complete!")
    print()
    print(f"📊 Statistics:")
    print(f"   PMC documents sampled: {len(selected_pmc_ids)}")
    print(f"   Valid image-annotation pairs: {sum(stats.values())}")
    print(f"   Images copied: {copied_stats['images']}")
    print(f"   Train annotations: {copied_stats['train']}")
    print(f"   Test annotations: {copied_stats['test']}")
    print(f"   Val annotations: {copied_stats['val']}")
    print(f"   Total files copied: {sum(copied_stats.values())}")
    
    if validation_errors:
        print(f"   ⚠️  Validation errors: {len(validation_errors)}")
    else:
        print(f"   ✅ No validation errors")
    
    print()
    
    # Verify image and annotation counts match
    annotation_total = copied_stats['train'] + copied_stats['test'] + copied_stats['val']
    if copied_stats['images'] == annotation_total:
        print("✅ Verification: Image and annotation counts match!")
    else:
        print(f"⚠️  Warning: Image count ({copied_stats['images']}) != Annotation count ({annotation_total})")
    
    print()
    print(f"📂 Output directory: {output_dir}")
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create a random subset of PubTables-1M dataset using filelists",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a 1000-document subset
  python scripts/pickup_data.py --num 1000 --output data/pubtables_subset_1k
  
  # Create a 10000-document subset
  python scripts/pickup_data.py --num 10000 --output data/pubtables_subset_10k
  
  # Use custom seed for reproducibility
  python scripts/pickup_data.py --num 5000 --seed 123 --output data/pubtables_subset_5k
  
  # Skip file verification (faster but risky)
  python scripts/pickup_data.py --num 1000 --no-verify --output data/pubtables_subset_1k
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
    
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip file existence verification (faster but less safe)"
    )
    
    args = parser.parse_args()
    
    # Validate source directory
    if not os.path.exists(args.source):
        print(f"❌ Error: Source directory not found: {args.source}")
        print(f"   Make sure you've extracted the dataset first.")
        return 1
    
    # Create subset
    create_subset(
        args.source, 
        args.output, 
        args.num, 
        args.seed,
        verify_files=not args.no_verify
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
