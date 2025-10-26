#!/usr/bin/env python3
"""
Generate metadata JSONL file for training the evaluation model.

This script processes all generated JSON files, compares them with ground truth,
and creates a JSONL file with filenames and similarity scores.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys
from tqdm import tqdm

# Import scoring functions from score_extraction.py
# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import after path is set
import importlib.util
spec = importlib.util.spec_from_file_location("score_extraction", parent_dir / "scripts" / "score_extraction.py")
score_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score_module)
load_cells_from_json = score_module.load_cells_from_json
evaluate_extraction = score_module.evaluate_extraction


def process_single_file(
    generated_file: Path,
    gt_file: Path,
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Process a single file pair and return metadata.
    
    Args:
        generated_file: Path to generated JSON
        gt_file: Path to ground truth JSON
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with id, filenames, and similarity_score
    """
    try:
        # Load cells
        generated_cells = load_cells_from_json(generated_file)
        gt_cells = load_cells_from_json(gt_file)
        
        # Evaluate
        scores = evaluate_extraction(generated_cells, gt_cells, iou_threshold)
        
        # Extract file identifier (without .json extension)
        file_id = generated_file.stem
        
        # Create metadata entry
        metadata = {
            "id": file_id,
            "ground_truth_file": gt_file.name,
            "generated_file": generated_file.name,
            "similarity_score": scores['overall_score']
        }
        
        return metadata
        
    except Exception as e:
        print(f"Error processing {generated_file.name}: {str(e)}")
        return None


def generate_metadata_jsonl(
    generated_dir: Path,
    gt_dir: Path,
    output_file: Path,
    iou_threshold: float = 0.5
) -> None:
    """
    Generate metadata JSONL file for all test files.
    
    Args:
        generated_dir: Directory containing generated JSON files
        gt_dir: Directory containing ground truth JSON files
        output_file: Output JSONL file path
        iou_threshold: IoU threshold for cell matching
    """
    # Find all generated JSON files
    generated_files = sorted(generated_dir.glob('*.json'))
    
    print(f"Found {len(generated_files)} generated files")
    print(f"IoU threshold: {iou_threshold}")
    
    # Statistics
    successful = 0
    failed = 0
    
    # Process files and write to JSONL
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for generated_file in tqdm(generated_files, desc="Processing files"):
            # Find matching ground truth file
            gt_file = gt_dir / generated_file.name
            
            if not gt_file.exists():
                print(f"Warning: No ground truth for {generated_file.name}")
                failed += 1
                continue
            
            # Process and get metadata
            metadata = process_single_file(generated_file, gt_file, iou_threshold)
            
            if metadata is None:
                failed += 1
                continue
            
            # Write as single-line JSON
            json_line = json.dumps(metadata, ensure_ascii=False)
            outfile.write(json_line + '\n')
            successful += 1
    
    # Print statistics
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Total files: {len(generated_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nOutput saved to: {output_file}")
    
    # Validate output
    if successful > 0:
        # Count lines
        line_count = sum(1 for _ in open(output_file))
        print(f"JSONL file contains {line_count} lines")
        
        # Show sample entry
        print("\nSample entry:")
        with open(output_file, 'r') as f:
            first_line = f.readline()
            sample = json.loads(first_line)
            print(json.dumps(sample, indent=2))


def analyze_metadata(output_file: Path) -> None:
    """Analyze the generated metadata JSONL file."""
    scores = []
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            scores.append(data['similarity_score'])
    
    if not scores:
        print("No scores to analyze!")
        return
    
    print("\n" + "="*70)
    print("SCORE ANALYSIS")
    print("="*70)
    print(f"Total files: {len(scores)}")
    print(f"Average score: {sum(scores)/len(scores):.3f}")
    print(f"Min score: {min(scores):.3f}")
    print(f"Max score: {max(scores):.3f}")
    
    # Score distribution
    buckets = [
        (0.0, 0.2, "Poor"),
        (0.2, 0.4, "Fair"),
        (0.4, 0.6, "Good"),
        (0.6, 0.8, "Very Good"),
        (0.8, 1.0, "Excellent")
    ]
    
    print("\nScore Distribution:")
    for min_val, max_val, label in buckets:
        count = sum(1 for s in scores if min_val <= s < max_val)
        pct = count / len(scores) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:12} ({min_val:.1f}-{max_val:.1f}): {count:4d} ({pct:5.1f}%) {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate metadata JSONL file for table extraction evaluation"
    )
    parser.add_argument(
        '--generated',
        type=Path,
        required=True,
        help='Directory containing generated JSON files'
    )
    parser.add_argument(
        '--ground-truth',
        type=Path,
        required=True,
        help='Directory containing ground truth JSON files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for cell matching (default: 0.5)'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze the generated metadata file'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.generated.exists():
        print(f"Error: Generated directory does not exist: {args.generated}")
        return 1
    
    if not args.ground_truth.exists():
        print(f"Error: Ground truth directory does not exist: {args.ground_truth}")
        return 1
    
    # Generate metadata
    generate_metadata_jsonl(
        args.generated,
        args.ground_truth,
        args.output,
        args.iou_threshold
    )
    
    # Analyze if requested
    if args.analyze:
        analyze_metadata(args.output)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

