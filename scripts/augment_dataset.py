#!/usr/bin/env python3
"""
Augment and balance the table extraction evaluation dataset.

This script creates a balanced dataset by:
1. Using existing samples from the Hugging Face dataset
2. Generating synthetic samples from SciTSR ground truth tables
3. Applying controlled perturbations to achieve target quality score ranges

The goal is to balance the distribution across all quality score ranges:
- 0.0-0.2: Very poor extraction (major structural errors)
- 0.2-0.4: Poor extraction (significant errors)
- 0.4-0.6: Moderate extraction (some errors)
- 0.6-0.8: Good extraction (minor errors)
- 0.8-1.0: Excellent extraction (nearly perfect)
"""

import argparse
import json
import random
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm


@dataclass
class Cell:
    """Represents a table cell with grid position and content."""
    id: int
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    content: List[str]
    tex: str = ""
    
    def grid_tuple(self) -> Tuple[int, int, int, int]:
        return (self.start_row, self.end_row, self.start_col, self.end_col)
    
    def grid_cells(self) -> Set[Tuple[int, int]]:
        cells = set()
        for row in range(self.start_row, self.end_row + 1):
            for col in range(self.start_col, self.end_col + 1):
                cells.add((row, col))
        return cells
    
    def text(self) -> str:
        return ' '.join(self.content)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "start_row": self.start_row,
            "end_row": self.end_row,
            "start_col": self.start_col,
            "end_col": self.end_col,
            "content": self.content,
            "tex": self.tex
        }


def load_cells_from_dict(data: Dict) -> List[Cell]:
    """Load cells from dictionary format."""
    cells = []
    for cell_data in data.get('cells', []):
        cell = Cell(
            id=cell_data.get('id', 0),
            start_row=cell_data.get('start_row', 0),
            end_row=cell_data.get('end_row', 0),
            start_col=cell_data.get('start_col', 0),
            end_col=cell_data.get('end_col', 0),
            content=cell_data.get('content', []),
            tex=cell_data.get('tex', "")
        )
        cells.append(cell)
    return cells


def cells_to_dict(cells: List[Cell]) -> Dict:
    """Convert cells list to dictionary format."""
    return {"cells": [cell.to_dict() for cell in cells]}


def calculate_grid_iou(cell1: Cell, cell2: Cell) -> float:
    """Calculate IoU of grid positions."""
    grid1 = cell1.grid_cells()
    grid2 = cell2.grid_cells()
    
    if not grid1 or not grid2:
        return 0.0
    
    intersection = len(grid1 & grid2)
    union = len(grid1 | grid2)
    
    return intersection / union if union > 0 else 0.0


def match_cells(
    pred_cells: List[Cell],
    gt_cells: List[Cell],
    iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match predicted cells with ground truth cells based on IoU."""
    matches = []
    matched_pred = set()
    matched_gt = set()
    
    iou_matrix = []
    for i, pred_cell in enumerate(pred_cells):
        for j, gt_cell in enumerate(gt_cells):
            iou = calculate_grid_iou(pred_cell, gt_cell)
            iou_matrix.append((iou, i, j))
    
    iou_matrix.sort(reverse=True, key=lambda x: x[0])
    
    for iou, pred_idx, gt_idx in iou_matrix:
        if iou < iou_threshold:
            break
        if pred_idx not in matched_pred and gt_idx not in matched_gt:
            matches.append((pred_idx, gt_idx))
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
    
    unmatched_pred = [i for i in range(len(pred_cells)) if i not in matched_pred]
    unmatched_gt = [i for i in range(len(gt_cells)) if i not in matched_gt]
    
    return matches, unmatched_pred, unmatched_gt


def calculate_similarity_score(pred_cells: List[Cell], gt_cells: List[Cell], iou_threshold: float = 0.5) -> float:
    """
    Calculate similarity score between predicted and ground truth cells.
    Uses the same formula as the original scoring script.
    """
    if not gt_cells:
        return 1.0 if not pred_cells else 0.0
    
    if not pred_cells:
        return 0.0
    
    matches, unmatched_pred, unmatched_gt = match_cells(pred_cells, gt_cells, iou_threshold)
    
    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    pred_rows = max([c.end_row for c in pred_cells], default=0) + 1
    pred_cols = max([c.end_col for c in pred_cells], default=0) + 1
    gt_rows = max([c.end_row for c in gt_cells], default=0) + 1
    gt_cols = max([c.end_col for c in gt_cells], default=0) + 1
    
    row_accuracy = 1.0 - abs(pred_rows - gt_rows) / max(pred_rows, gt_rows, 1)
    col_accuracy = 1.0 - abs(pred_cols - gt_cols) / max(pred_cols, gt_cols, 1)
    
    overall_score = 0.50 * f1 + 0.25 * row_accuracy + 0.25 * col_accuracy
    
    return overall_score


def perturb_table_severe(gt_cells: List[Cell], intensity: float = 0.9) -> List[Cell]:
    """
    Apply severe perturbations for very low quality scores (0.0-0.2).
    - Delete most cells
    - Major grid shifts
    - Completely wrong structure
    """
    if not gt_cells:
        return []
    
    cells = copy.deepcopy(gt_cells)
    
    # Delete 70-90% of cells
    delete_ratio = 0.7 + random.random() * 0.2 * intensity
    num_to_keep = max(1, int(len(cells) * (1 - delete_ratio)))
    cells = random.sample(cells, num_to_keep)
    
    # Major grid shifts
    max_shift = 5
    for cell in cells:
        if random.random() < 0.8:
            shift = random.randint(-max_shift, max_shift)
            cell.start_row = max(0, cell.start_row + shift)
            cell.end_row = max(cell.start_row, cell.end_row + shift)
        if random.random() < 0.8:
            shift = random.randint(-max_shift, max_shift)
            cell.start_col = max(0, cell.start_col + shift)
            cell.end_col = max(cell.start_col, cell.end_col + shift)
    
    # Add random garbage cells
    if random.random() < 0.5:
        for i in range(random.randint(3, 8)):
            garbage_cell = Cell(
                id=1000 + i,
                start_row=random.randint(0, 20),
                end_row=random.randint(0, 20),
                start_col=random.randint(0, 20),
                end_col=random.randint(0, 20),
                content=["garbage", str(random.randint(0, 100))],
                tex=""
            )
            # Ensure end >= start
            garbage_cell.end_row = max(garbage_cell.start_row, garbage_cell.end_row)
            garbage_cell.end_col = max(garbage_cell.start_col, garbage_cell.end_col)
            cells.append(garbage_cell)
    
    return cells


def perturb_table_heavy(gt_cells: List[Cell], intensity: float = 0.7) -> List[Cell]:
    """
    Apply heavy perturbations for poor quality scores (0.2-0.4).
    - Delete 40-60% of cells
    - Moderate grid shifts
    - Some wrong cells added
    """
    if not gt_cells:
        return []
    
    cells = copy.deepcopy(gt_cells)
    
    # Delete 40-60% of cells
    delete_ratio = 0.4 + random.random() * 0.2 * intensity
    num_to_keep = max(1, int(len(cells) * (1 - delete_ratio)))
    cells = random.sample(cells, num_to_keep)
    
    # Moderate grid shifts
    max_shift = 3
    for cell in cells:
        if random.random() < 0.5:
            shift = random.randint(-max_shift, max_shift)
            cell.start_row = max(0, cell.start_row + shift)
            cell.end_row = max(cell.start_row, cell.end_row + shift)
        if random.random() < 0.5:
            shift = random.randint(-max_shift, max_shift)
            cell.start_col = max(0, cell.start_col + shift)
            cell.end_col = max(cell.start_col, cell.end_col + shift)
    
    # Sometimes merge adjacent cells incorrectly
    if len(cells) > 2 and random.random() < 0.4:
        idx = random.randint(0, len(cells) - 2)
        cells[idx].end_row = max(cells[idx].end_row, cells[idx + 1].end_row)
        cells[idx].end_col = max(cells[idx].end_col, cells[idx + 1].end_col)
        cells.pop(idx + 1)
    
    # Add some random cells
    if random.random() < 0.3:
        for i in range(random.randint(1, 3)):
            gt_rows = max([c.end_row for c in gt_cells], default=5)
            gt_cols = max([c.end_col for c in gt_cells], default=5)
            random_cell = Cell(
                id=500 + i,
                start_row=random.randint(0, gt_rows + 2),
                end_row=random.randint(0, gt_rows + 2),
                start_col=random.randint(0, gt_cols + 2),
                end_col=random.randint(0, gt_cols + 2),
                content=["extra"],
                tex=""
            )
            random_cell.end_row = max(random_cell.start_row, random_cell.end_row)
            random_cell.end_col = max(random_cell.start_col, random_cell.end_col)
            cells.append(random_cell)
    
    return cells


def perturb_table_moderate(gt_cells: List[Cell], intensity: float = 0.5) -> List[Cell]:
    """
    Apply moderate perturbations for medium quality scores (0.4-0.6).
    - Delete 20-35% of cells
    - Minor grid shifts
    - Occasional wrong merges
    """
    if not gt_cells:
        return []
    
    cells = copy.deepcopy(gt_cells)
    
    # Delete 20-35% of cells
    delete_ratio = 0.2 + random.random() * 0.15 * intensity
    num_to_keep = max(1, int(len(cells) * (1 - delete_ratio)))
    cells = random.sample(cells, num_to_keep)
    
    # Minor grid shifts
    for cell in cells:
        if random.random() < 0.3:
            shift = random.choice([-2, -1, 1, 2])
            cell.start_row = max(0, cell.start_row + shift)
            cell.end_row = max(cell.start_row, cell.end_row + shift)
        if random.random() < 0.3:
            shift = random.choice([-2, -1, 1, 2])
            cell.start_col = max(0, cell.start_col + shift)
            cell.end_col = max(cell.start_col, cell.end_col + shift)
    
    return cells


def perturb_table_light(gt_cells: List[Cell], intensity: float = 0.3) -> List[Cell]:
    """
    Apply light perturbations for good quality scores (0.6-0.8).
    - Delete 5-15% of cells
    - Very minor grid shifts
    """
    if not gt_cells:
        return []
    
    cells = copy.deepcopy(gt_cells)
    
    # Delete 5-15% of cells
    delete_ratio = 0.05 + random.random() * 0.10 * intensity
    num_to_keep = max(1, int(len(cells) * (1 - delete_ratio)))
    cells = random.sample(cells, num_to_keep)
    
    # Very minor grid shifts
    for cell in cells:
        if random.random() < 0.15:
            shift = random.choice([-1, 1])
            cell.start_row = max(0, cell.start_row + shift)
            cell.end_row = max(cell.start_row, cell.end_row + shift)
        if random.random() < 0.15:
            shift = random.choice([-1, 1])
            cell.start_col = max(0, cell.start_col + shift)
            cell.end_col = max(cell.start_col, cell.end_col + shift)
    
    return cells


def perturb_table_minimal(gt_cells: List[Cell], intensity: float = 0.1) -> List[Cell]:
    """
    Apply minimal perturbations for excellent quality scores (0.8-1.0).
    - Delete 0-5% of cells
    - Rare minor shifts
    """
    if not gt_cells:
        return []
    
    cells = copy.deepcopy(gt_cells)
    
    # Delete 0-5% of cells
    delete_ratio = random.random() * 0.05 * intensity
    if delete_ratio > 0 and len(cells) > 1:
        num_to_keep = max(1, int(len(cells) * (1 - delete_ratio)))
        cells = random.sample(cells, num_to_keep)
    
    # Very rare grid shifts
    for cell in cells:
        if random.random() < 0.05:
            shift = random.choice([-1, 1])
            cell.start_row = max(0, cell.start_row + shift)
            cell.end_row = max(cell.start_row, cell.end_row + shift)
    
    return cells


def generate_sample_with_target_score(
    gt_dict: Dict,
    target_range: Tuple[float, float],
    max_attempts: int = 50
) -> Tuple[Dict, float]:
    """
    Generate a perturbed sample targeting a specific score range.
    
    Returns:
        Tuple of (generated_dict, actual_score)
    """
    gt_cells = load_cells_from_dict(gt_dict)
    
    if not gt_cells:
        return None, 0.0
    
    target_low, target_high = target_range
    target_mid = (target_low + target_high) / 2
    
    # Select perturbation function based on target range
    if target_high <= 0.2:
        perturb_func = perturb_table_severe
    elif target_high <= 0.4:
        perturb_func = perturb_table_heavy
    elif target_high <= 0.6:
        perturb_func = perturb_table_moderate
    elif target_high <= 0.8:
        perturb_func = perturb_table_light
    else:
        perturb_func = perturb_table_minimal
    
    best_cells = None
    best_score = -1
    best_diff = float('inf')
    
    for attempt in range(max_attempts):
        intensity = 0.3 + random.random() * 0.7
        perturbed_cells = perturb_func(gt_cells, intensity)
        score = calculate_similarity_score(perturbed_cells, gt_cells)
        
        diff = abs(score - target_mid)
        
        # Check if score is in target range
        if target_low <= score <= target_high:
            return cells_to_dict(perturbed_cells), score
        
        # Keep track of best attempt
        if diff < best_diff:
            best_diff = diff
            best_cells = perturbed_cells
            best_score = score
    
    # Return best attempt even if not in range
    if best_cells:
        return cells_to_dict(best_cells), best_score
    
    return None, 0.0


def load_scitsr_tables(scitsr_dir: Path, split: str = 'train', limit: int = None) -> List[Dict]:
    """Load ground truth tables from SciTSR dataset."""
    structure_dir = scitsr_dir / split / 'structure'
    
    if not structure_dir.exists():
        print(f"Warning: SciTSR structure directory not found: {structure_dir}")
        return []
    
    tables = []
    json_files = sorted(structure_dir.glob('*.json'))
    
    if limit:
        json_files = json_files[:limit]
    
    for json_file in tqdm(json_files, desc=f"Loading SciTSR {split}"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data['_source_file'] = json_file.name
            tables.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return tables


def load_huggingface_dataset(split: str = 'train', limit: int = None) -> List[Dict]:
    """Load existing samples from Hugging Face dataset."""
    try:
        from datasets import load_dataset
        
        print(f"Loading {split} split from Hugging Face...")
        dataset = load_dataset("rayhu/table-extraction-evaluation", split=split)
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        samples = []
        for example in tqdm(dataset, desc=f"Loading HF {split}"):
            samples.append({
                'id': example['id'],
                'similarity_score': example['similarity_score'],
                'ground_truth': example['ground_truth'],
                'generated': example['generated']
            })
        
        return samples
        
    except Exception as e:
        print(f"Failed to load from Hugging Face: {e}")
        return []


def analyze_distribution(samples: List[Dict]) -> Dict[str, int]:
    """Analyze the score distribution of samples."""
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    bin_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    
    counts = {label: 0 for label in bin_labels}
    
    for sample in samples:
        score = sample['similarity_score']
        for (low, high), label in zip(bins, bin_labels):
            if low <= score < high or (label == '0.8-1.0' and score == 1.0):
                counts[label] += 1
                break
    
    return counts


def create_balanced_dataset(
    hf_samples: List[Dict],
    scitsr_tables: List[Dict],
    target_per_bin: int = 5000,
    output_dir: Path = None
) -> List[Dict]:
    """
    Create a balanced dataset by augmenting underrepresented ranges.
    
    Args:
        hf_samples: Existing samples from Hugging Face
        scitsr_tables: Ground truth tables from SciTSR
        target_per_bin: Target number of samples per score range
        output_dir: Directory to save augmented samples
    
    Returns:
        List of all samples (existing + augmented)
    """
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    bin_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    
    # Categorize existing samples
    binned_samples = {label: [] for label in bin_labels}
    
    for sample in hf_samples:
        score = sample['similarity_score']
        for (low, high), label in zip(bins, bin_labels):
            if low <= score < high or (label == '0.8-1.0' and score == 1.0):
                binned_samples[label].append(sample)
                break
    
    print("\n📊 Initial Distribution:")
    for label in bin_labels:
        count = len(binned_samples[label])
        print(f"  {label}: {count:5d} samples")
    
    # Calculate how many samples to generate for each bin
    to_generate = {}
    for label in bin_labels:
        current = len(binned_samples[label])
        needed = max(0, target_per_bin - current)
        to_generate[label] = needed
    
    print("\n🎯 Samples to Generate:")
    for label in bin_labels:
        print(f"  {label}: {to_generate[label]:5d} needed")
    
    # Generate synthetic samples
    augmented_samples = []
    scitsr_idx = 0
    
    for (low, high), label in zip(bins, bin_labels):
        needed = to_generate[label]
        if needed <= 0:
            continue
        
        print(f"\n🔄 Generating {needed} samples for range {label}...")
        
        generated_count = 0
        attempts = 0
        max_total_attempts = needed * 10  # Allow multiple attempts per sample
        
        pbar = tqdm(total=needed, desc=f"  {label}")
        
        while generated_count < needed and attempts < max_total_attempts:
            # Cycle through SciTSR tables
            gt_table = scitsr_tables[scitsr_idx % len(scitsr_tables)]
            scitsr_idx += 1
            attempts += 1
            
            generated_dict, actual_score = generate_sample_with_target_score(
                gt_table, (low, high), max_attempts=30
            )
            
            if generated_dict is None:
                continue
            
            # Check if score is in target range
            if low <= actual_score < high or (label == '0.8-1.0' and actual_score >= 0.8):
                sample = {
                    'id': f"aug_{label}_{generated_count}",
                    'similarity_score': actual_score,
                    'ground_truth': gt_table,
                    'generated': generated_dict,
                    '_augmented': True,
                    '_source': gt_table.get('_source_file', 'unknown')
                }
                augmented_samples.append(sample)
                binned_samples[label].append(sample)
                generated_count += 1
                pbar.update(1)
        
        pbar.close()
        
        if generated_count < needed:
            print(f"  ⚠️  Only generated {generated_count}/{needed} samples for {label}")
    
    # Combine all samples
    all_samples = hf_samples + augmented_samples
    
    print("\n📊 Final Distribution:")
    final_dist = analyze_distribution(all_samples)
    for label in bin_labels:
        print(f"  {label}: {final_dist[label]:5d} samples")
    
    print(f"\n✅ Total samples: {len(all_samples)}")
    print(f"   Original: {len(hf_samples)}")
    print(f"   Augmented: {len(augmented_samples)}")
    
    # Save augmented samples if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save augmented samples
        aug_file = output_dir / 'augmented_samples.json'
        with open(aug_file, 'w') as f:
            json.dump(augmented_samples, f)
        print(f"\n💾 Augmented samples saved to: {aug_file}")
        
        # Save distribution info
        dist_file = output_dir / 'augmented_distribution.json'
        dist_info = {
            'original_count': len(hf_samples),
            'augmented_count': len(augmented_samples),
            'total_count': len(all_samples),
            'target_per_bin': target_per_bin,
            'final_distribution': final_dist
        }
        with open(dist_file, 'w') as f:
            json.dump(dist_info, f, indent=2)
        print(f"💾 Distribution info saved to: {dist_file}")
    
    return all_samples


def plot_distributions(original_dist: Dict, augmented_dist: Dict, output_path: Path):
    """Plot comparison of original and augmented distributions."""
    import matplotlib.pyplot as plt
    
    labels = list(original_dist.keys())
    original_counts = [original_dist[l] for l in labels]
    augmented_counts = [augmented_dist[l] for l in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width/2, original_counts, width, label='Original', 
                   color='#FF6B6B', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, augmented_counts, width, label='Augmented',
                   color='#4ECDC4', edgecolor='black', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 100,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Quality Score Range', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Distribution: Original vs Augmented', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Distribution plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Augment and balance the table extraction evaluation dataset"
    )
    parser.add_argument(
        '--scitsr-dir',
        type=Path,
        default=Path('SciTSR'),
        help='Path to SciTSR dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data_augmented'),
        help='Directory to save augmented data'
    )
    parser.add_argument(
        '--target-per-bin',
        type=int,
        default=5000,
        help='Target number of samples per score range (default: 5000)'
    )
    parser.add_argument(
        '--hf-limit',
        type=int,
        default=None,
        help='Limit on Hugging Face samples to load (for testing)'
    )
    parser.add_argument(
        '--scitsr-limit',
        type=int,
        default=None,
        help='Limit on SciTSR tables to load (for testing)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("="*70)
    print("TABLE EXTRACTION DATASET AUGMENTATION")
    print("="*70)
    
    # Load existing data
    print("\n📥 Loading existing dataset...")
    hf_samples = load_huggingface_dataset('train', limit=args.hf_limit)
    
    if not hf_samples:
        print("❌ Failed to load Hugging Face dataset. Exiting.")
        return 1
    
    # Analyze original distribution
    original_dist = analyze_distribution(hf_samples)
    
    # Load SciTSR tables
    print("\n📥 Loading SciTSR ground truth tables...")
    scitsr_tables = load_scitsr_tables(args.scitsr_dir, 'train', limit=args.scitsr_limit)
    
    if not scitsr_tables:
        print("❌ Failed to load SciTSR tables. Exiting.")
        return 1
    
    print(f"   Loaded {len(scitsr_tables)} SciTSR tables")
    
    # Create balanced dataset
    print("\n🔄 Creating balanced dataset...")
    balanced_samples = create_balanced_dataset(
        hf_samples,
        scitsr_tables,
        target_per_bin=args.target_per_bin,
        output_dir=args.output_dir
    )
    
    # Plot distributions
    augmented_dist = analyze_distribution(balanced_samples)
    plot_path = args.output_dir / 'distribution_comparison.png'
    plot_distributions(original_dist, augmented_dist, plot_path)
    
    print("\n" + "="*70)
    print("✅ AUGMENTATION COMPLETE!")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
