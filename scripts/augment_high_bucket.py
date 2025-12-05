#!/usr/bin/env python3
"""
Focused augmentation script for High bucket (0.6-0.8) only.

This script specifically targets the 0.6-0.8 score range with improved accuracy
by using iterative refinement to get scores closer to the target range.
"""

import argparse
import json
import random
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import scoring functions
import importlib.util
spec = importlib.util.spec_from_file_location(
    "score_extraction", 
    Path(__file__).parent.parent / "scripts" / "score_extraction.py"
)
score_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score_module)
load_cells_from_json = score_module.load_cells_from_json
evaluate_extraction = score_module.evaluate_extraction
Cell = score_module.Cell


def calculate_score(gt_cells: List[Dict], generated_cells: List[Dict]) -> float:
    """Calculate similarity score between GT and generated cells."""
    gt_cell_objects = [
        Cell(
            id=c.get('id', 0),
            start_row=c.get('start_row', 0),
            end_row=c.get('end_row', 0),
            start_col=c.get('start_col', 0),
            end_col=c.get('end_col', 0),
            content=c.get('content', [])
        )
        for c in gt_cells
    ]
    gen_cell_objects = [
        Cell(
            id=c.get('id', 0),
            start_row=c.get('start_row', 0),
            end_row=c.get('end_row', 0),
            start_col=c.get('start_col', 0),
            end_col=c.get('end_col', 0),
            content=c.get('content', [])
        )
        for c in generated_cells
    ]
    scores = evaluate_extraction(gen_cell_objects, gt_cell_objects)
    return scores['overall_score']


def align_cell_towards_gt(cell: Dict, gt_cell: Dict, alignment_factor: float) -> None:
    """Align a cell partially towards ground truth position."""
    cell['start_row'] = int(
        cell['start_row'] * (1 - alignment_factor) + 
        gt_cell['start_row'] * alignment_factor
    )
    cell['end_row'] = int(
        cell['end_row'] * (1 - alignment_factor) + 
        gt_cell['end_row'] * alignment_factor
    )
    cell['start_col'] = int(
        cell['start_col'] * (1 - alignment_factor) + 
        gt_cell['start_col'] * alignment_factor
    )
    cell['end_col'] = int(
        cell['end_col'] * (1 - alignment_factor) + 
        gt_cell['end_col'] * alignment_factor
    )


def create_high_score_variant_iterative(
    gt_cells: List[Dict], 
    generated_cells: List[Dict],
    target_min: float = 0.6,
    target_max: float = 0.8,
    max_iterations: int = 10
) -> Tuple[List[Dict], float]:
    """
    Create a variant in the 0.6-0.8 range using iterative refinement.
    
    Strategy:
    1. Start from generated cells (which are usually lower scores)
    2. Iteratively fix cells by aligning them towards GT
    3. Check score after each iteration
    4. Stop when we hit the target range or max iterations
    """
    if not gt_cells or not generated_cells:
        return copy.deepcopy(generated_cells), 0.0
    
    modified = copy.deepcopy(generated_cells)
    current_score = calculate_score(gt_cells, modified)
    
    # If already in range, apply small random variation
    if target_min <= current_score < target_max:
        # Apply very small random shifts to create variation
        for _ in range(random.randint(1, 2)):
            if modified:
                cell = random.choice(modified)
                cell['start_row'] = max(0, cell['start_row'] + random.randint(-1, 1))
                cell['start_col'] = max(0, cell['start_col'] + random.randint(-1, 1))
        return modified, current_score
    
    # If score is too low, fix cells iteratively
    if current_score < target_min:
        best_modified = copy.deepcopy(modified)
        best_score = current_score
        
        for iteration in range(max_iterations):
            # Calculate how much we need to improve
            score_gap = target_min - current_score
            # More aggressive fixes if we're far from target
            alignment_strength = min(0.9, 0.3 + score_gap * 0.5)
            
            # Fix a few cells each iteration
            num_fixes = min(3, max(1, len(modified) // 5))
            fixed_indices = set()
            
            for _ in range(num_fixes):
                if not modified or not gt_cells:
                    break
                
                # Find a cell to fix
                mod_idx = random.randint(0, len(modified) - 1)
                if mod_idx in fixed_indices:
                    continue
                fixed_indices.add(mod_idx)
                
                # Find closest GT cell to align with
                mod_cell = modified[mod_idx]
                best_gt_idx = 0
                min_distance = float('inf')
                
                for gt_idx, gt_cell in enumerate(gt_cells):
                    # Calculate distance based on position
                    row_dist = abs(mod_cell['start_row'] - gt_cell['start_row']) + \
                              abs(mod_cell['end_row'] - gt_cell['end_row'])
                    col_dist = abs(mod_cell['start_col'] - gt_cell['start_col']) + \
                              abs(mod_cell['end_col'] - gt_cell['end_col'])
                    distance = row_dist + col_dist
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_gt_idx = gt_idx
                
                # Align towards GT
                align_cell_towards_gt(modified[mod_idx], gt_cells[best_gt_idx], alignment_strength)
            
            # Check new score
            new_score = calculate_score(gt_cells, modified)
            
            # If we improved and are closer to target, keep it
            if new_score > current_score:
                best_modified = copy.deepcopy(modified)
                best_score = new_score
                current_score = new_score
            
            # If we're in range, we're done
            if target_min <= new_score < target_max:
                return modified, new_score
            
            # If we overshot, back off slightly
            if new_score >= target_max:
                # Use the best version we had before overshooting
                if best_score < target_max:
                    return best_modified, best_score
                # Otherwise, apply small random perturbations to reduce score
                for cell in modified:
                    if random.random() < 0.3:
                        cell['start_row'] = max(0, cell['start_row'] + random.randint(-1, 1))
                        cell['start_col'] = max(0, cell['start_col'] + random.randint(-1, 1))
                new_score = calculate_score(gt_cells, modified)
                if target_min <= new_score < target_max:
                    return modified, new_score
        
        # Return best we found
        return best_modified, best_score
    
    # If score is too high, introduce small errors
    else:  # current_score >= target_max
        # Apply small random shifts to reduce score slightly
        for _ in range(random.randint(2, 4)):
            if modified:
                cell = random.choice(modified)
                cell['start_row'] = max(0, cell['start_row'] + random.randint(-1, 1))
                cell['end_row'] = max(cell['start_row'], cell['end_row'] + random.randint(-1, 1))
                cell['start_col'] = max(0, cell['start_col'] + random.randint(-1, 1))
                cell['end_col'] = max(cell['start_col'], cell['end_col'] + random.randint(-1, 1))
        
        new_score = calculate_score(gt_cells, modified)
        if target_min <= new_score < target_max:
            return modified, new_score
        
        # If still too high, try removing a few cells
        if len(modified) > 2:
            num_to_remove = min(2, len(modified) // 10)
            indices_to_remove = random.sample(range(len(modified)), num_to_remove)
            modified = [c for i, c in enumerate(modified) if i not in indices_to_remove]
            new_score = calculate_score(gt_cells, modified)
        
        return modified, new_score


def augment_high_bucket(
    metadata_file: Path,
    generated_dir: Path,
    gt_dir: Path,
    output_metadata: Path,
    target_count: int = 6000,
    target_min: float = 0.6,
    target_max: float = 0.8
) -> None:
    """
    Augment only the High bucket (0.6-0.8 range).
    
    Args:
        metadata_file: Input metadata JSONL file
        generated_dir: Directory with generated JSON files
        gt_dir: Directory with ground truth JSON files
        output_metadata: Output metadata JSONL file
        target_count: Target number of samples in High bucket
        target_min: Minimum score for High bucket (default: 0.6)
        target_max: Maximum score for High bucket (default: 0.8)
    """
    print("="*70)
    print("HIGH BUCKET AUGMENTATION (0.6-0.8)")
    print("="*70)
    
    # Load all samples
    print("\nLoading samples...")
    all_samples = []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            all_samples.append(json.loads(line))
    
    # Count current High bucket samples
    high_samples = [s for s in all_samples if target_min <= s['similarity_score'] < target_max]
    current_count = len(high_samples)
    
    print(f"\nCurrent High bucket (0.6-0.8): {current_count} samples")
    print(f"Target: {target_count} samples")
    print(f"Need to generate: {max(0, target_count - current_count)} samples")
    
    if current_count >= target_count:
        print("\n✓ Already have enough samples in High bucket!")
        return
    
    needed = target_count - current_count
    
    # Use all samples as candidates (better diversity)
    candidates = all_samples
    print(f"\nUsing {len(candidates)} candidate samples for augmentation...")
    
    # Generate augmented samples
    print(f"\nGenerating {needed} samples in High bucket (0.6-0.8)...")
    new_samples = []
    successful = 0
    failed = 0
    
    # Create augmented directory
    augmented_dir = generated_dir / 'augmented'
    augmented_dir.mkdir(parents=True, exist_ok=True)
    
    for i in tqdm(range(needed), desc="Augmenting"):
        # Try up to 5 times to get a sample in the target range
        for attempt in range(5):
            base_sample = random.choice(candidates)
            
            try:
                # Load files
                gt_file = gt_dir / base_sample['ground_truth_file']
                generated_file_path = base_sample['generated_file']
                
                # Try to find generated file
                generated_file = generated_dir / generated_file_path
                if not generated_file.exists():
                    generated_file = generated_dir / 'augmented' / Path(generated_file_path).name
                if not generated_file.exists():
                    generated_file = generated_dir / Path(generated_file_path).name
                
                if not gt_file.exists() or not generated_file.exists():
                    failed += 1
                    continue
                
                with open(gt_file, 'r') as f:
                    gt_data = json.load(f)
                with open(generated_file, 'r') as f:
                    generated_data = json.load(f)
                
                gt_cells = gt_data.get('cells', [])
                generated_cells = generated_data.get('cells', [])
                
                if not gt_cells or not generated_cells:
                    failed += 1
                    continue
                
                # Create high-score variant
                modified_cells, new_score = create_high_score_variant_iterative(
                    gt_cells, generated_cells, target_min, target_max
                )
                
                # Check if score is in target range
                if target_min <= new_score < target_max:
                    # Create new file
                    new_id = f"{base_sample['id']}_high_{random.randint(10000, 99999)}"
                    new_generated_data = {'cells': modified_cells}
                    new_generated_file = augmented_dir / f"{new_id}.json"
                    
                    try:
                        with open(new_generated_file, 'w') as f:
                            json.dump(new_generated_data, f, indent=2)
                    except Exception as e:
                        print(f"\nWarning: Could not write {new_generated_file}: {e}")
                        continue
                    
                    # Create metadata entry
                    new_metadata = {
                        'id': new_id,
                        'ground_truth_file': base_sample['ground_truth_file'],
                        'generated_file': f"augmented/{new_id}.json",
                        'similarity_score': new_score,
                        'augmented_from': base_sample['id'],
                        'augmentation_type': 'high'
                    }
                    
                    new_samples.append(new_metadata)
                    successful += 1
                    break  # Success, move to next sample
                else:
                    # Score not in range, try again
                    if attempt == 4:  # Last attempt
                        failed += 1
                    continue
                    
            except Exception as e:
                if attempt == 4:
                    failed += 1
                continue
    
    print(f"\n✓ Generated {successful} new samples")
    print(f"✗ Failed: {failed}")
    
    # Write output
    print(f"\nWriting augmented dataset to {output_metadata}...")
    with open(output_metadata, 'w', encoding='utf-8') as f:
        # Write all original samples
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # Write new augmented samples
        for sample in new_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Analyze final distribution
    print("\n" + "="*70)
    print("FINAL DISTRIBUTION")
    print("="*70)
    
    final_samples = all_samples + new_samples
    high_final = [s for s in final_samples if target_min <= s['similarity_score'] < target_max]
    
    print(f"Total samples: {len(final_samples)}")
    print(f"High bucket (0.6-0.8): {len(high_final)} samples")
    print(f"  - Original: {current_count}")
    print(f"  - New: {len(high_final) - current_count}")
    print(f"  - Target: {target_count}")
    
    if len(high_final) >= target_count:
        print("\n✓ Successfully reached target!")
    else:
        print(f"\n⚠ Still need {target_count - len(high_final)} more samples")


def main():
    parser = argparse.ArgumentParser(
        description="Augment only the High bucket (0.6-0.8) with improved accuracy"
    )
    parser.add_argument(
        '--metadata',
        type=Path,
        required=True,
        help='Input metadata JSONL file'
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
        help='Output metadata JSONL file'
    )
    parser.add_argument(
        '--target-count',
        type=int,
        default=6000,
        help='Target number of samples in High bucket (default: 6000)'
    )
    parser.add_argument(
        '--target-min',
        type=float,
        default=0.6,
        help='Minimum score for High bucket (default: 0.6)'
    )
    parser.add_argument(
        '--target-max',
        type=float,
        default=0.8,
        help='Maximum score for High bucket (default: 0.8)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.metadata.exists():
        print(f"Error: Metadata file does not exist: {args.metadata}")
        return 1
    
    if not args.generated.exists():
        print(f"Error: Generated directory does not exist: {args.generated}")
        return 1
    
    if not args.ground_truth.exists():
        print(f"Error: Ground truth directory does not exist: {args.ground_truth}")
        return 1
    
    # Run augmentation
    augment_high_bucket(
        args.metadata,
        args.generated,
        args.ground_truth,
        args.output,
        args.target_count,
        args.target_min,
        args.target_max
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())



