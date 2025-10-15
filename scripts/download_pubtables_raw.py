#!/usr/bin/env python3
"""
Download raw PubTables-1M tar.gz files from Hugging Face.
These are the original files that can be extracted using extract_structure_dataset.sh
"""

import os
from huggingface_hub import hf_hub_download

# List of all raw data files available in the PubTables-1M repository
RAW_FILES = {
    "structure": [
        "PubTables-1M-Structure_Filelists.tar.gz",
        "PubTables-1M-Structure_Annotations_Test.tar.gz",
        "PubTables-1M-Structure_Annotations_Train.tar.gz",
        "PubTables-1M-Structure_Annotations_Val.tar.gz",
        "PubTables-1M-Structure_Images_Test.tar.gz",
        "PubTables-1M-Structure_Images_Train.tar.gz",
        "PubTables-1M-Structure_Images_Val.tar.gz",
        "PubTables-1M-Structure_Table_Words.tar.gz",
    ],
    "detection": [
        "PubTables-1M-Detection_Filelists.tar.gz",
        "PubTables-1M-Detection_Annotations_Test.tar.gz",
        "PubTables-1M-Detection_Annotations_Train.tar.gz",
        "PubTables-1M-Detection_Annotations_Val.tar.gz",
        "PubTables-1M-Detection_Images_Test.tar.gz",
        "PubTables-1M-Detection_Images_Train_Part1.tar.gz",
        "PubTables-1M-Detection_Images_Train_Part2.tar.gz",
        "PubTables-1M-Detection_Images_Val.tar.gz",
        "PubTables-1M-Detection_Page_Words.tar.gz",
    ],
    "pdf": [
        "PubTables-1M-PDF_Annotations.tar.gz",
    ]
}

# File sizes (approximate, in GB)
FILE_SIZES = {
    "PubTables-1M-Structure_Images_Train.tar.gz": 20.3,
    "PubTables-1M-Detection_Images_Train_Part1.tar.gz": 28.7,
    "PubTables-1M-Detection_Images_Train_Part2.tar.gz": 27.3,
    "PubTables-1M-Detection_Images_Test.tar.gz": 6.9,
    "PubTables-1M-Detection_Images_Val.tar.gz": 7.0,
    "PubTables-1M-Detection_Page_Words.tar.gz": 6.0,
    "PubTables-1M-PDF_Annotations.tar.gz": 3.4,
    "PubTables-1M-Structure_Table_Words.tar.gz": 3.9,
    "PubTables-1M-Structure_Images_Test.tar.gz": 2.5,
    "PubTables-1M-Structure_Images_Val.tar.gz": 2.5,
}

def download_raw_files(output_dir="data/pubtables_raw", dataset_type="all"):
    """
    Download raw PubTables-1M tar.gz files.
    
    Args:
        output_dir: Directory to save the files
        dataset_type: "structure", "detection", "pdf", or "all"
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which files to download
    if dataset_type == "all":
        files_to_download = (
            RAW_FILES["structure"] + 
            RAW_FILES["detection"] + 
            RAW_FILES["pdf"]
        )
    elif dataset_type in RAW_FILES:
        files_to_download = RAW_FILES[dataset_type]
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be 'structure', 'detection', 'pdf', or 'all'")
    
    # Calculate total size
    total_size_gb = sum(FILE_SIZES.get(f, 0.1) for f in files_to_download)
    
    print(f"📥 PubTables-1M Raw Data Download")
    print(f"=" * 50)
    print(f"Dataset type: {dataset_type}")
    print(f"Files to download: {len(files_to_download)}")
    print(f"Approximate total size: {total_size_gb:.1f} GB")
    print(f"Output directory: {output_dir}")
    print(f"=" * 50)
    print()
    
    # Download each file
    skipped = 0
    downloaded = 0
    failed = 0
    
    for i, filename in enumerate(files_to_download, 1):
        size_info = f" (~{FILE_SIZES[filename]:.1f} GB)" if filename in FILE_SIZES else ""
        target_path = os.path.join(output_dir, filename)
        
        # Check if file already exists and is not empty
        if os.path.exists(target_path):
            file_size = os.path.getsize(target_path)
            if file_size > 0:
                file_size_mb = file_size / (1024 * 1024)
                if file_size_mb > 1024:
                    size_str = f"{file_size_mb / 1024:.2f} GB"
                else:
                    size_str = f"{file_size_mb:.2f} MB"
                print(f"[{i}/{len(files_to_download)}] ⏭️  Skipping {filename} (already downloaded, {size_str})")
                skipped += 1
                continue
            else:
                print(f"[{i}/{len(files_to_download)}] ⚠️  Found empty file, re-downloading {filename}{size_info}...")
                os.remove(target_path)
        else:
            print(f"[{i}/{len(files_to_download)}] Downloading {filename}{size_info}...")
        
        try:
            downloaded_path = hf_hub_download(
                repo_id="bsmock/pubtables-1m",
                filename=filename,
                repo_type="dataset",
                cache_dir=None,  # Use default cache
                local_dir=output_dir,
                local_dir_use_symlinks=False,  # Copy files instead of symlinks
            )
            print(f"    ✅ Downloaded to: {downloaded_path}")
            downloaded += 1
        except Exception as e:
            print(f"    ❌ Error downloading {filename}: {e}")
            print(f"    Continuing with next file...")
            failed += 1
        print()
    
    print("=" * 50)
    print("✅ Download session complete!")
    print()
    print(f"📊 Summary:")
    print(f"   ✅ Downloaded: {downloaded} file(s)")
    print(f"   ⏭️  Skipped (already exists): {skipped} file(s)")
    print(f"   ❌ Failed: {failed} file(s)")
    print(f"   📂 Files saved to: {output_dir}")
    print()
    
    if failed > 0:
        print("⚠️  Some files failed to download.")
        print("   This is often due to disk space issues.")
        print("   Fix the issue and re-run this script - it will skip already downloaded files.")
        print()
    
    if downloaded > 0 or (skipped > 0 and failed == 0):
        print("Next steps:")
        print("1. Navigate to the data directory: cd data/pubtables_raw")
        print("2. Run extraction script: bash ../../scripts/extract_structure_dataset.sh")
        print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download raw PubTables-1M tar.gz files from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all files (Structure + Detection + PDF) - ~110 GB
  python scripts/download_pubtables_raw_raw.py --type all
  
  # Download only Structure dataset files - ~30 GB
  python scripts/download_pubtables_raw_raw.py --type structure
  
  # Download only Detection dataset files - ~77 GB
  python scripts/download_pubtables_raw_raw.py --type detection
  
  # Download to custom directory
  python scripts/download_pubtables_raw_raw.py --type structure --output data/my_folder
        """
    )
    
    parser.add_argument(
        "--output",
        default="data/pubtables_raw",
        help="Output directory (default: data/pubtables_raw)"
    )
    
    parser.add_argument(
        "--type",
        choices=["structure", "detection", "pdf", "all"],
        default="structure",
        help="Dataset type to download (default: structure)"
    )
    
    args = parser.parse_args()
    
    download_raw_files(args.output, args.type)

