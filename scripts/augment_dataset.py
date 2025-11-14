#!/usr/bin/env python3
"""
Data augmentation script for table extraction evaluation dataset.

This script addresses the class imbalance problem where most similarity scores
are concentrated in the 0.4-0.6 range by:
1. Oversampling underrepresented score ranges
2. Creating structural variations of existing tables
3. Generating synthetic table pairs with controlled similarity scores
"""

import argparse
import json
import random
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
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


def analyze_score_distribution(metadata_file: Path) -> Dict[str, Any]:
    """Analyze the current score distribution."""
    scores = []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            scores.append(data['similarity_score'])
    
    if not scores:
        return {}
    
    buckets = [
        (0.0, 0.2, "Very Low"),
        (0.2, 0.4, "Low"),
        (0.4, 0.6, "Medium"),
        (0.6, 0.8, "High"),
        (0.8, 1.0, "Very High")
    ]
    
    distribution = {}
    for min_val, max_val, label in buckets:
        count = sum(1 for s in scores if min_val <= s < max_val)
        distribution[label] = {
            'count': count,
            'percentage': count / len(scores) * 100,
            'range': (min_val, max_val)
        }
    
    return {
        'total': len(scores),
        'mean': sum(scores) / len(scores),
        'min': min(scores),
        'max': max(scores),
        'distribution': distribution,
        'all_scores': scores
    }


def modify_cell_positions(cells: List[Dict], modification_type: str = 'shift') -> List[Dict]:
    """
    Modify cell positions to create structural variations.
    
    Args:
        cells: List of cell dictionaries
        modification_type: 'shift', 'merge', 'split', or 'remove'
    
    Returns:
        Modified list of cells
    """
    cells = copy.deepcopy(cells)
    
    if not cells:
        return cells
    
    if modification_type == 'shift':
        # Randomly shift some cells' positions
        num_to_shift = max(1, len(cells) // 10)  # Shift ~10% of cells
        cells_to_shift = random.sample(cells, min(num_to_shift, len(cells)))
        
        for cell in cells_to_shift:
            # Small random shift (0-2 positions)
            row_shift = random.randint(-1, 1)
            col_shift = random.randint(-1, 1)
            cell['start_row'] = max(0, cell['start_row'] + row_shift)
            cell['end_row'] = max(cell['start_row'], cell['end_row'] + row_shift)
            cell['start_col'] = max(0, cell['start_col'] + col_shift)
            cell['end_col'] = max(cell['start_col'], cell['end_col'] + col_shift)
    
    elif modification_type == 'merge':
        # Merge adjacent cells
        if len(cells) < 2:
            return cells
        
        # Find cells that can be merged (same row or same col, adjacent)
        candidates = []
        for i, cell1 in enumerate(cells):
            for j, cell2 in enumerate(cells[i+1:], i+1):
                # Same row, adjacent columns
                if (cell1['start_row'] == cell2['start_row'] == cell1['end_row'] == cell2['end_row'] and
                    abs(cell1['end_col'] - cell2['start_col']) <= 1):
                    candidates.append((i, j))
                # Same column, adjacent rows
                elif (cell1['start_col'] == cell2['start_col'] == cell1['end_col'] == cell2['end_col'] and
                      abs(cell1['end_row'] - cell2['start_row']) <= 1):
                    candidates.append((i, j))
        
        if candidates:
            i, j = random.choice(candidates)
            cell1, cell2 = cells[i], cells[j]
            # Merge: combine content and extend boundaries
            merged_cell = {
                'id': cell1['id'],
                'tex': cell1.get('tex', ''),
                'content': cell1.get('content', []) + cell2.get('content', []),
                'start_row': min(cell1['start_row'], cell2['start_row']),
                'end_row': max(cell1['end_row'], cell2['end_row']),
                'start_col': min(cell1['start_col'], cell2['start_col']),
                'end_col': max(cell1['end_col'], cell2['end_col'])
            }
            cells[i] = merged_cell
            cells.pop(j)
    
    elif modification_type == 'remove':
        # Remove some cells (but keep at least 50%)
        num_to_remove = random.randint(1, max(1, len(cells) // 2))
        if num_to_remove < len(cells):
            cells_to_remove = random.sample(range(len(cells)), num_to_remove)
            cells = [c for i, c in enumerate(cells) if i not in cells_to_remove]
    
    return cells


def create_low_score_variant(gt_cells: List[Dict], generated_cells: List[Dict]) -> Tuple[List[Dict], float]:
    """
    Create a variant with lower similarity score by introducing more errors.
    
    Returns:
        Modified generated cells and expected score range
    """
    modified = copy.deepcopy(generated_cells)
    
    # Apply multiple modifications to reduce similarity
    modifications = ['shift', 'remove', 'merge']
    num_mods = random.randint(2, 4)
    
    for _ in range(num_mods):
        mod_type = random.choice(modifications)
        modified = modify_cell_positions(modified, mod_type)
    
    return modified, (0.0, 0.4)  # Target range: low scores


def create_high_score_variant(gt_cells: List[Dict], generated_cells: List[Dict]) -> Tuple[List[Dict], float]:
    """
    Create a variant with higher similarity score by fixing errors.
    
    Returns:
        Modified generated cells and expected score range
    """
    # Start with ground truth and introduce small variations
    modified = copy.deepcopy(gt_cells)
    
    # Apply minimal modifications (small shifts only)
    num_mods = random.randint(1, 2)
    for _ in range(num_mods):
        modified = modify_cell_positions(modified, 'shift')
    
    return modified, (0.6, 1.0)  # Target range: high scores


def create_medium_score_variant(gt_cells: List[Dict], generated_cells: List[Dict]) -> Tuple[List[Dict], float]:
    """
    Create a variant with medium similarity score.
    
    Returns:
        Modified generated cells and expected score range
    """
    # Mix of ground truth and generated, with some modifications
    modified = copy.deepcopy(generated_cells)
    
    # Apply moderate modifications
    num_mods = random.randint(1, 2)
    for _ in range(num_mods):
        mod_type = random.choice(['shift', 'merge'])
        modified = modify_cell_positions(modified, mod_type)
    
    return modified, (0.4, 0.6)  # Target range: medium scores


def augment_sample(
    metadata_entry: Dict,
    generated_dir: Path,
    gt_dir: Path,
    target_score_range: Optional[Tuple[float, float]] = None,
    variant_type: Optional[str] = None
) -> Optional[Dict]:
    """
    Create an augmented version of a sample.
    
    Args:
        metadata_entry: Original metadata entry
        generated_dir: Directory with generated JSON files
        gt_dir: Directory with ground truth JSON files
        target_score_range: Desired score range (min, max)
        variant_type: 'low', 'high', 'medium', or None (random)
    
    Returns:
        New metadata entry or None if failed
    """
    try:
        # Load original files
        gt_file = gt_dir / metadata_entry['ground_truth_file']
        generated_file_path = metadata_entry['generated_file']
        
        # Try to find the generated file in various locations
        generated_file = generated_dir / generated_file_path
        if not generated_file.exists():
            # Try augmented subdirectory
            generated_file = generated_dir / 'augmented' / Path(generated_file_path).name
        if not generated_file.exists():
            # Try original location (without augmented prefix)
            generated_file = generated_dir / Path(generated_file_path).name
        
        if not gt_file.exists() or not generated_file.exists():
            return None
        
        with open(gt_file, 'r') as f:
            gt_data = json.load(f)
        with open(generated_file, 'r') as f:
            generated_data = json.load(f)
        
        gt_cells = gt_data.get('cells', [])
        generated_cells = generated_data.get('cells', [])
        
        # Determine variant type if not specified
        if variant_type is None:
            if target_score_range:
                if target_score_range[1] < 0.4:
                    variant_type = 'low'
                elif target_score_range[0] > 0.6:
                    variant_type = 'high'
                else:
                    variant_type = 'medium'
            else:
                variant_type = random.choice(['low', 'medium', 'high'])
        
        # Create variant
        if variant_type == 'low':
            modified_cells, expected_range = create_low_score_variant(gt_cells, generated_cells)
        elif variant_type == 'high':
            modified_cells, expected_range = create_high_score_variant(gt_cells, generated_cells)
        else:  # medium
            modified_cells, expected_range = create_medium_score_variant(gt_cells, generated_cells)
        
        # Calculate actual score
        # Only pass fields that Cell accepts (ignore 'tex' and other extra fields)
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
        modified_cell_objects = [
            Cell(
                id=c.get('id', 0),
                start_row=c.get('start_row', 0),
                end_row=c.get('end_row', 0),
                start_col=c.get('start_col', 0),
                end_col=c.get('end_col', 0),
                content=c.get('content', [])
            )
            for c in modified_cells
        ]
        scores = evaluate_extraction(modified_cell_objects, gt_cell_objects)
        new_score = scores['overall_score']
        
        # Check if score is in desired range
        if target_score_range and not (target_score_range[0] <= new_score < target_score_range[1]):
            # Try once more with different modifications
            if variant_type == 'low':
                modified_cells, _ = create_low_score_variant(gt_cells, generated_cells)
            elif variant_type == 'high':
                modified_cells, _ = create_high_score_variant(gt_cells, generated_cells)
            else:
                modified_cells, _ = create_medium_score_variant(gt_cells, generated_cells)
            
            modified_cell_objects = [
                Cell(
                    id=c.get('id', 0),
                    start_row=c.get('start_row', 0),
                    end_row=c.get('end_row', 0),
                    start_col=c.get('start_col', 0),
                    end_col=c.get('end_col', 0),
                    content=c.get('content', [])
                )
                for c in modified_cells
            ]
            scores = evaluate_extraction(modified_cell_objects, gt_cell_objects)
            new_score = scores['overall_score']
        
        # Create new file
        # Use a separate 'augmented' subdirectory to avoid permission issues
        augmented_dir = generated_dir / 'augmented'
        augmented_dir.mkdir(parents=True, exist_ok=True)
        
        new_id = f"{metadata_entry['id']}_aug_{variant_type}_{random.randint(1000, 9999)}"
        new_generated_data = {'cells': modified_cells}
        new_generated_file = augmented_dir / f"{new_id}.json"
        
        try:
            with open(new_generated_file, 'w') as f:
                json.dump(new_generated_data, f, indent=2)
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not write to {new_generated_file}: {e}")
            print(f"Trying alternative location...")
            # Try writing to a temp directory or current directory
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / 'table_augmentation'
            temp_dir.mkdir(parents=True, exist_ok=True)
            new_generated_file = temp_dir / f"{new_id}.json"
            with open(new_generated_file, 'w') as f:
                json.dump(new_generated_data, f, indent=2)
            print(f"Wrote to temporary location: {new_generated_file}")
        
        # Create new metadata entry
        # Store relative path from generated_dir
        if 'augmented' in str(new_generated_file):
            # File is in augmented subdirectory
            relative_path = f"augmented/{new_id}.json"
        else:
            # File is in temp directory or elsewhere
            relative_path = str(new_generated_file.name)
        
        new_metadata = {
            'id': new_id,
            'ground_truth_file': metadata_entry['ground_truth_file'],
            'generated_file': relative_path,
            'similarity_score': new_score,
            'augmented_from': metadata_entry['id'],
            'augmentation_type': variant_type
        }
        
        return new_metadata
        
    except Exception as e:
        print(f"Error augmenting {metadata_entry.get('id', 'unknown')}: {str(e)}")
        return None


def oversample_range(
    metadata_file: Path,
    generated_dir: Path,
    gt_dir: Path,
    target_range: Tuple[float, float],
    target_count: int,
    variant_type: str
) -> List[Dict]:
    """
    Oversample a specific score range.
    
    Args:
        metadata_file: Original metadata file
        target_range: (min_score, max_score) to target
        target_count: Number of samples to generate
        variant_type: 'low', 'high', or 'medium'
    
    Returns:
        List of new metadata entries
    """
    # Load all samples
    all_samples = []
    with open(metadata_file, 'r') as f:
        for line in f:
            all_samples.append(json.loads(line))
    
    # Filter samples in target range (or nearby)
    candidates = [s for s in all_samples if target_range[0] <= s['similarity_score'] < target_range[1]]
    
    # If not enough candidates, use all samples
    if len(candidates) < target_count:
        candidates = all_samples
    
    # Generate augmented samples
    new_samples = []
    for _ in tqdm(range(target_count), desc=f"Augmenting {variant_type} range"):
        base_sample = random.choice(candidates)
        augmented = augment_sample(
            base_sample,
            generated_dir,
            gt_dir,
            target_score_range=target_range,
            variant_type=variant_type
        )
        if augmented:
            new_samples.append(augmented)
    
    return new_samples


def augment_dataset(
    metadata_file: Path,
    generated_dir: Path,
    gt_dir: Path,
    output_metadata: Path,
    target_distribution: Optional[Dict[str, int]] = None,
    augmentation_factor: float = 1.0
) -> None:
    """
    Augment the dataset to balance score distribution.
    
    Args:
        metadata_file: Input metadata JSONL file
        generated_dir: Directory with generated JSON files
        gt_dir: Directory with ground truth JSON files
        output_metadata: Output metadata JSONL file
        target_distribution: Target counts per range (optional)
        augmentation_factor: Multiplier for augmentation (1.0 = balance to current max)
    """
    # Analyze current distribution
    print("Analyzing current distribution...")
    analysis = analyze_score_distribution(metadata_file)
    
    if not analysis:
        print("Error: Could not analyze distribution")
        return
    
    print("\nCurrent Distribution:")
    print(f"Total samples: {analysis['total']}")
    print(f"Mean score: {analysis['mean']:.3f}")
    print(f"Score range: {analysis['min']:.3f} - {analysis['max']:.3f}")
    print("\nScore buckets:")
    for label, info in analysis['distribution'].items():
        print(f"  {label:12} ({info['range'][0]:.1f}-{info['range'][1]:.1f}): "
              f"{info['count']:4d} ({info['percentage']:5.1f}%)")
    
    # Determine target counts
    if target_distribution is None:
        # Balance to the maximum count in any bucket
        max_count = max(info['count'] for info in analysis['distribution'].values())
        target_distribution = {
            'Very Low': int(max_count * augmentation_factor * 0.3),  # Less for extremes
            'Low': int(max_count * augmentation_factor * 0.7),
            'Medium': int(max_count * augmentation_factor),  # Keep current
            'High': int(max_count * augmentation_factor * 0.7),
            'Very High': int(max_count * augmentation_factor * 0.3)
        }
    
    print("\nTarget Distribution:")
    for label, target_count in target_distribution.items():
        current = analysis['distribution'][label]['count']
        needed = max(0, target_count - current)
        print(f"  {label:12}: Current={current:4d}, Target={target_count:4d}, Need={needed:4d}")
    
    # Load original samples
    print("\nLoading original samples...")
    original_samples = []
    with open(metadata_file, 'r') as f:
        for line in f:
            original_samples.append(json.loads(line))
    
    # Generate augmented samples
    print("\nGenerating augmented samples...")
    all_new_samples = []
    
    ranges = {
        'Very Low': ((0.0, 0.2), 'low'),
        'Low': ((0.2, 0.4), 'low'),
        'High': ((0.6, 0.8), 'high'),
        'Very High': ((0.8, 1.0), 'high')
    }
    
    for label, (score_range, variant_type) in ranges.items():
        current_count = analysis['distribution'][label]['count']
        target_count = target_distribution[label]
        needed = max(0, target_count - current_count)
        
        if needed > 0:
            print(f"\nAugmenting {label} range ({score_range[0]:.1f}-{score_range[1]:.1f})...")
            new_samples = oversample_range(
                metadata_file,
                generated_dir,
                gt_dir,
                score_range,
                needed,
                variant_type
            )
            all_new_samples.extend(new_samples)
            print(f"  Generated {len(new_samples)} new samples")
    
    # Write output
    print(f"\nWriting augmented dataset to {output_metadata}...")
    with open(output_metadata, 'w', encoding='utf-8') as f:
        # Write original samples
        for sample in original_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # Write augmented samples
        for sample in all_new_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Analyze final distribution
    print("\nFinal Distribution:")
    final_analysis = analyze_score_distribution(output_metadata)
    print(f"Total samples: {final_analysis['total']} (original: {analysis['total']}, new: {len(all_new_samples)})")
    print(f"Mean score: {final_analysis['mean']:.3f}")
    print("\nFinal score buckets:")
    for label, info in final_analysis['distribution'].items():
        print(f"  {label:12} ({info['range'][0]:.1f}-{info['range'][1]:.1f}): "
              f"{info['count']:4d} ({info['percentage']:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Augment table extraction evaluation dataset to balance score distribution"
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
        '--augmentation-factor',
        type=float,
        default=1.0,
        help='Augmentation factor (1.0 = balance to max, 2.0 = double max, etc.)'
    )
    parser.add_argument(
        '--target-very-low',
        type=int,
        help='Target count for Very Low (0.0-0.2) range'
    )
    parser.add_argument(
        '--target-low',
        type=int,
        help='Target count for Low (0.2-0.4) range'
    )
    parser.add_argument(
        '--target-medium',
        type=int,
        help='Target count for Medium (0.4-0.6) range'
    )
    parser.add_argument(
        '--target-high',
        type=int,
        help='Target count for High (0.6-0.8) range'
    )
    parser.add_argument(
        '--target-very-high',
        type=int,
        help='Target count for Very High (0.8-1.0) range'
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
    
    # Build target distribution if specified
    target_dist = None
    if any([args.target_very_low, args.target_low, args.target_medium, 
            args.target_high, args.target_very_high]):
        target_dist = {
            'Very Low': args.target_very_low or 0,
            'Low': args.target_low or 0,
            'Medium': args.target_medium or 0,
            'High': args.target_high or 0,
            'Very High': args.target_very_high or 0
        }
    
    # Run augmentation
    augment_dataset(
        args.metadata,
        args.generated,
        args.ground_truth,
        args.output,
        target_distribution=target_dist,
        augmentation_factor=args.augmentation_factor
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

