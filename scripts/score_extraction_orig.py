#!/usr/bin/env python3
"""
Score table extraction quality by comparing with ground truth.

This script implements multiple evaluation metrics for table structure extraction:
- Cell-level Precision, Recall, F1 (based on grid IoU)
- Structure accuracy (row/column detection)
- Content accuracy (text matching)
- Overall quality scores
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@dataclass
class Cell:
    """Represents a table cell with grid position and content."""
    id: int
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    content: List[str]
    
    def grid_tuple(self) -> Tuple[int, int, int, int]:
        """Return grid position as tuple for comparison."""
        return (self.start_row, self.end_row, self.start_col, self.end_col)
    
    def grid_cells(self) -> Set[Tuple[int, int]]:
        """Return set of (row, col) tuples this cell occupies."""
        cells = set()
        for row in range(self.start_row, self.end_row + 1):
            for col in range(self.start_col, self.end_col + 1):
                cells.add((row, col))
        return cells
    
    def text(self) -> str:
        """Return cell text as single string."""
        return ' '.join(self.content)


def load_cells_from_json(json_path: Path) -> List[Cell]:
    """Load cells from SciTSR JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cells = []
    for cell_data in data.get('cells', []):
        cell = Cell(
            id=cell_data.get('id', 0),
            start_row=cell_data.get('start_row', 0),
            end_row=cell_data.get('end_row', 0),
            start_col=cell_data.get('start_col', 0),
            end_col=cell_data.get('end_col', 0),
            content=cell_data.get('content', [])
        )
        cells.append(cell)
    
    return cells


def calculate_grid_iou(cell1: Cell, cell2: Cell) -> float:
    """
    Calculate IoU (Intersection over Union) of grid positions.
    
    IoU = |intersection| / |union| of grid cells occupied
    """
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
    """
    Match predicted cells with ground truth cells based on IoU.
    
    Returns:
        - matches: List of (pred_idx, gt_idx) pairs
        - unmatched_pred: Indices of unmatched predicted cells
        - unmatched_gt: Indices of unmatched ground truth cells
    """
    matches = []
    matched_pred = set()
    matched_gt = set()
    
    # Create IoU matrix
    iou_matrix = []
    for i, pred_cell in enumerate(pred_cells):
        row = []
        for j, gt_cell in enumerate(gt_cells):
            iou = calculate_grid_iou(pred_cell, gt_cell)
            row.append((iou, i, j))
        iou_matrix.extend(row)
    
    # Sort by IoU descending
    iou_matrix.sort(reverse=True, key=lambda x: x[0])
    
    # Greedy matching
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


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate text similarity using word-level Jaccard similarity.
    
    Returns value between 0 and 1.
    """
    if not text1 and not text2:
        return 1.0
    
    if not text1 or not text2:
        return 0.0
    
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def calculate_exact_match(text1: str, text2: str) -> bool:
    """Check if texts match exactly (case-insensitive, whitespace-normalized)."""
    t1 = ' '.join(text1.lower().split())
    t2 = ' '.join(text2.lower().split())
    return t1 == t2


def evaluate_extraction(
    pred_cells: List[Cell],
    gt_cells: List[Cell],
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate extraction quality with multiple metrics.
    
    Returns dictionary with scores and statistics.
    """
    # Match cells
    matches, unmatched_pred, unmatched_gt = match_cells(
        pred_cells, gt_cells, iou_threshold
    )
    
    # Cell-level metrics
    tp = len(matches)  # True positives
    fp = len(unmatched_pred)  # False positives
    fn = len(unmatched_gt)  # False negatives
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Content accuracy for matched cells
    text_similarities = []
    exact_matches = 0
    
    for pred_idx, gt_idx in matches:
        pred_text = pred_cells[pred_idx].text()
        gt_text = gt_cells[gt_idx].text()
        
        similarity = calculate_text_similarity(pred_text, gt_text)
        text_similarities.append(similarity)
        
        if calculate_exact_match(pred_text, gt_text):
            exact_matches += 1
    
    avg_text_similarity = sum(text_similarities) / len(text_similarities) if text_similarities else 0.0
    exact_match_rate = exact_matches / len(matches) if matches else 0.0
    
    # Structure accuracy
    pred_rows = max([c.end_row for c in pred_cells], default=0) + 1
    pred_cols = max([c.end_col for c in pred_cells], default=0) + 1
    gt_rows = max([c.end_row for c in gt_cells], default=0) + 1
    gt_cols = max([c.end_col for c in gt_cells], default=0) + 1
    
    row_accuracy = 1.0 - abs(pred_rows - gt_rows) / max(pred_rows, gt_rows, 1)
    col_accuracy = 1.0 - abs(pred_cols - gt_cols) / max(pred_cols, gt_cols, 1)
    
    # Overall score (weighted combination)
    overall_score = (
        0.4 * f1 +  # Cell detection is most important
        0.3 * avg_text_similarity +  # Text accuracy
        0.15 * row_accuracy +  # Structure accuracy
        0.15 * col_accuracy
    )
    
    return {
        # Cell detection metrics
        'cell_detection': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        },
        
        # Content accuracy
        'content_accuracy': {
            'avg_text_similarity': avg_text_similarity,
            'exact_match_rate': exact_match_rate,
            'exact_matches': exact_matches,
            'total_matched': len(matches)
        },
        
        # Structure accuracy
        'structure_accuracy': {
            'row_accuracy': row_accuracy,
            'col_accuracy': col_accuracy,
            'pred_rows': pred_rows,
            'pred_cols': pred_cols,
            'gt_rows': gt_rows,
            'gt_cols': gt_cols
        },
        
        # Cell counts
        'cell_counts': {
            'predicted': len(pred_cells),
            'ground_truth': len(gt_cells),
            'difference': len(pred_cells) - len(gt_cells)
        },
        
        # Overall score
        'overall_score': overall_score,
        
        # Matching details
        'matching': {
            'iou_threshold': iou_threshold,
            'matched_pairs': len(matches),
            'unmatched_predicted': len(unmatched_pred),
            'unmatched_ground_truth': len(unmatched_gt)
        }
    }


def print_evaluation_report(scores: Dict[str, Any], detailed: bool = False) -> None:
    """Print formatted evaluation report."""
    print("\n" + "="*70)
    print("TABLE EXTRACTION EVALUATION REPORT")
    print("="*70)
    
    # Overall score
    print(f"\n📊 OVERALL SCORE: {scores['overall_score']:.3f} ({scores['overall_score']*100:.1f}%)")
    
    # Cell detection
    cd = scores['cell_detection']
    print(f"\n🔍 Cell Detection:")
    print(f"  Precision: {cd['precision']:.3f} ({cd['precision']*100:.1f}%)")
    print(f"  Recall:    {cd['recall']:.3f} ({cd['recall']*100:.1f}%)")
    print(f"  F1 Score:  {cd['f1']:.3f} ({cd['f1']*100:.1f}%)")
    print(f"  TP: {cd['true_positives']}, FP: {cd['false_positives']}, FN: {cd['false_negatives']}")
    
    # Content accuracy
    ca = scores['content_accuracy']
    print(f"\n📝 Content Accuracy:")
    print(f"  Avg Text Similarity: {ca['avg_text_similarity']:.3f} ({ca['avg_text_similarity']*100:.1f}%)")
    print(f"  Exact Match Rate:    {ca['exact_match_rate']:.3f} ({ca['exact_match_rate']*100:.1f}%)")
    print(f"  Exact Matches: {ca['exact_matches']}/{ca['total_matched']}")
    
    # Structure accuracy
    sa = scores['structure_accuracy']
    print(f"\n🏗️  Structure Accuracy:")
    print(f"  Row Accuracy: {sa['row_accuracy']:.3f} ({sa['row_accuracy']*100:.1f}%)")
    print(f"  Col Accuracy: {sa['col_accuracy']:.3f} ({sa['col_accuracy']*100:.1f}%)")
    print(f"  Predicted: {sa['pred_rows']}×{sa['pred_cols']} | Ground Truth: {sa['gt_rows']}×{sa['gt_cols']}")
    
    # Cell counts
    cc = scores['cell_counts']
    print(f"\n📦 Cell Counts:")
    print(f"  Predicted: {cc['predicted']}")
    print(f"  Ground Truth: {cc['ground_truth']}")
    print(f"  Difference: {cc['difference']:+d}")
    
    if detailed:
        print(f"\n🔗 Matching Details:")
        m = scores['matching']
        print(f"  IoU Threshold: {m['iou_threshold']}")
        print(f"  Matched Pairs: {m['matched_pairs']}")
        print(f"  Unmatched Predicted: {m['unmatched_predicted']}")
        print(f"  Unmatched Ground Truth: {m['unmatched_ground_truth']}")
    
    print("\n" + "="*70)


def compare_files(
    pred_file: Path,
    gt_file: Path,
    iou_threshold: float = 0.5,
    detailed: bool = False
) -> Dict[str, Any]:
    """Compare two JSON files and return evaluation scores."""
    # Load cells
    pred_cells = load_cells_from_json(pred_file)
    gt_cells = load_cells_from_json(gt_file)
    
    # Evaluate
    scores = evaluate_extraction(pred_cells, gt_cells, iou_threshold)
    
    return scores


def batch_evaluate(
    pred_dir: Path,
    gt_dir: Path,
    iou_threshold: float = 0.5,
    output_file: Path = None
) -> Dict[str, Any]:
    """
    Evaluate all files in prediction directory against ground truth.
    
    Returns aggregated statistics.
    """
    pred_files = sorted(pred_dir.glob('*.json'))
    pred_files = [f for f in pred_files if f.name != 'processing_stats.json']
    
    all_scores = []
    failed_files = []
    
    print(f"Evaluating {len(pred_files)} files...")
    
    for pred_file in pred_files:
        gt_file = gt_dir / pred_file.name
        
        if not gt_file.exists():
            print(f"Warning: No ground truth for {pred_file.name}")
            continue
        
        try:
            scores = compare_files(pred_file, gt_file, iou_threshold)
            scores['filename'] = pred_file.name
            all_scores.append(scores)
        except Exception as e:
            print(f"Error processing {pred_file.name}: {str(e)}")
            failed_files.append(pred_file.name)
    
    if not all_scores:
        print("No files successfully evaluated!")
        return {}
    
    # Aggregate statistics
    aggregated = {
        'num_files': len(all_scores),
        'failed_files': len(failed_files),
        'iou_threshold': iou_threshold,
        'averages': {},
        'ranges': {},
        'distribution': {}
    }
    
    # Calculate averages for each metric
    metrics = [
        ('overall_score', lambda s: s['overall_score']),
        ('cell_precision', lambda s: s['cell_detection']['precision']),
        ('cell_recall', lambda s: s['cell_detection']['recall']),
        ('cell_f1', lambda s: s['cell_detection']['f1']),
        ('text_similarity', lambda s: s['content_accuracy']['avg_text_similarity']),
        ('exact_match_rate', lambda s: s['content_accuracy']['exact_match_rate']),
        ('row_accuracy', lambda s: s['structure_accuracy']['row_accuracy']),
        ('col_accuracy', lambda s: s['structure_accuracy']['col_accuracy'])
    ]
    
    for metric_name, extractor in metrics:
        values = [extractor(s) for s in all_scores]
        aggregated['averages'][metric_name] = sum(values) / len(values)
        aggregated['ranges'][metric_name] = {
            'min': min(values),
            'max': max(values),
            'std': (sum((v - aggregated['averages'][metric_name])**2 for v in values) / len(values))**0.5
        }
    
    # Score distribution
    score_buckets = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bucket_counts = {f"{score_buckets[i]:.1f}-{score_buckets[i+1]:.1f}": 0 
                     for i in range(len(score_buckets)-1)}
    
    for scores in all_scores:
        score = scores['overall_score']
        for i in range(len(score_buckets)-1):
            if score_buckets[i] <= score <= score_buckets[i+1]:
                bucket_counts[f"{score_buckets[i]:.1f}-{score_buckets[i+1]:.1f}"] += 1
                break
    
    aggregated['distribution'] = bucket_counts
    aggregated['individual_scores'] = all_scores
    
    # Save if requested
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(aggregated, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")
    
    return aggregated


def print_batch_report(aggregated: Dict[str, Any]) -> None:
    """Print aggregated evaluation report."""
    print("\n" + "="*70)
    print("BATCH EVALUATION REPORT")
    print("="*70)
    
    print(f"\nFiles Evaluated: {aggregated['num_files']}")
    if aggregated['failed_files'] > 0:
        print(f"Failed: {aggregated['failed_files']}")
    print(f"IoU Threshold: {aggregated['iou_threshold']}")
    
    print("\n📊 Average Scores:")
    avgs = aggregated['averages']
    ranges = aggregated['ranges']
    
    print(f"  Overall Score:     {avgs['overall_score']:.3f} ± {ranges['overall_score']['std']:.3f}")
    print(f"  Cell F1:           {avgs['cell_f1']:.3f} ± {ranges['cell_f1']['std']:.3f}")
    print(f"  Cell Precision:    {avgs['cell_precision']:.3f} ± {ranges['cell_precision']['std']:.3f}")
    print(f"  Cell Recall:       {avgs['cell_recall']:.3f} ± {ranges['cell_recall']['std']:.3f}")
    print(f"  Text Similarity:   {avgs['text_similarity']:.3f} ± {ranges['text_similarity']['std']:.3f}")
    print(f"  Exact Match Rate:  {avgs['exact_match_rate']:.3f} ± {ranges['exact_match_rate']['std']:.3f}")
    print(f"  Row Accuracy:      {avgs['row_accuracy']:.3f} ± {ranges['row_accuracy']['std']:.3f}")
    print(f"  Col Accuracy:      {avgs['col_accuracy']:.3f} ± {ranges['col_accuracy']['std']:.3f}")
    
    print("\n📈 Score Distribution:")
    for bucket, count in aggregated['distribution'].items():
        pct = count / aggregated['num_files'] * 100
        bar = "█" * int(pct / 2)
        print(f"  {bucket}: {count:4d} ({pct:5.1f}%) {bar}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Score table extraction quality against ground truth"
    )
    parser.add_argument(
        '--pred',
        type=Path,
        required=True,
        help='Predicted JSON file or directory'
    )
    parser.add_argument(
        '--gt',
        type=Path,
        required=True,
        help='Ground truth JSON file or directory'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for cell matching (default: 0.5)'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed matching information'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Save detailed scores to JSON file'
    )
    
    args = parser.parse_args()
    
    # Check if paths exist
    if not args.pred.exists():
        print(f"Error: Prediction path does not exist: {args.pred}")
        return 1
    
    if not args.gt.exists():
        print(f"Error: Ground truth path does not exist: {args.gt}")
        return 1
    
    # Single file comparison
    if args.pred.is_file() and args.gt.is_file():
        print(f"Comparing:\n  Pred: {args.pred}\n  GT:   {args.gt}")
        scores = compare_files(args.pred, args.gt, args.iou_threshold, args.detailed)
        print_evaluation_report(scores, args.detailed)
        
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(scores, f, indent=2)
            print(f"\nScores saved to: {args.output}")
    
    # Batch directory comparison
    elif args.pred.is_dir() and args.gt.is_dir():
        aggregated = batch_evaluate(args.pred, args.gt, args.iou_threshold, args.output)
        
        if aggregated:
            print_batch_report(aggregated)
    
    else:
        print("Error: Both paths must be either files or directories")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

