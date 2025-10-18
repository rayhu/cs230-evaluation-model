#!/usr/bin/env python3
"""
Re-download corrupted tar.gz file
"""

import os
from huggingface_hub import hf_hub_download

def redownload_file(filename, output_dir="data/pubtables_raw"):
    """Re-download a specific file from PubTables-1M"""
    
    print(f"📥 Re-downloading {filename}...")
    print(f"   Output: {output_dir}")
    print()
    
    # Check disk space first
    import shutil
    stat = shutil.disk_usage(output_dir)
    free_gb = stat.free / (1024**3)
    
    print(f"💾 Available disk space: {free_gb:.1f} GB")
    
    if free_gb < 25:
        print(f"⚠️  Warning: Low disk space! Need at least 25 GB")
        print(f"   (20 GB for file + 5 GB buffer)")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    print()
    print("Starting download... This will take a while (~20 GB)")
    print()
    
    try:
        downloaded_path = hf_hub_download(
            repo_id="bsmock/pubtables-1m",
            filename=filename,
            repo_type="dataset",
            cache_dir=None,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            resume_download=True,  # Resume if interrupted
        )
        print()
        print(f"✅ Downloaded successfully!")
        print(f"   Location: {downloaded_path}")
        print()
        print("Next step: Run verification script to confirm:")
        print("   bash scripts/verify_tar_integrity.sh")
        
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # Default to the corrupted file
        filename = "PubTables-1M-Structure_Images_Train.tar.gz"
        print("No filename provided, using default corrupted file:")
        print(f"  {filename}")
        print()
    
    exit(redownload_file(filename))


