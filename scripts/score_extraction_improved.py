#!/usr/bin/env python3
"""
Improved table extraction scoring with more tolerant matching.

Key improvements over the original:
1. Text-based fallback matching for unmatched cells
2. Fuzzy text matching with edit distance
3. Better text normalization (punctuation, special chars)
4. Multiple matching strategies (strict, moderate, lenient)
"""

import argparse
import json
import re
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
    
    def normalized_text(self) -> str:
        """Return normalized text for fuzzy matching."""
        text = self.text().lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove punctuation except numbers and basic symbols
        text = re.sub(r'[^\w\s\-\+\.\,]', '', text)
        return text


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


def calculate_levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return calculate_levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_text_similarity_fuzzy(text1: str, text2: str) -> float:
    """
    Calculate fuzzy text similarity using multiple methods.
    
    Combines:
    - Word-level Jaccard similarity
    - Character-level edit distance
    - Normalized length similarity
    
    Returns value between 0 and 1.
    """
    if not text1 and not text2:
        return 1.0
    
    if not text1 or not text2:
        return 0.0
    
    # Normalize text
    t1 = ' '.join(text1.lower().split())
    t2 = ' '.join(text2.lower().split())
    
    # Remove punctuation for comparison
    t1_clean = re.sub(r'[^\w\s]', '', t1)
    t2_clean = re.sub(r'[^\w\s]', '', t2)
    
    # Word-level Jaccard similarity
    words1 = set(t1_clean.split())
    words2 = set(t2_clean.split())
    
    if words1 or words2:
        word_intersection = len(words1 & words2)
        word_union = len(words1 | words2)
        jaccard_sim = word_intersection / word_union if word_union > 0 else 0.0
    else:
        jaccard_sim = 1.0 if not t1_clean and not t2_clean else 0.0
    
    # Character-level edit distance similarity
    if t1_clean or t2_clean:
        max_len = max(len(t1_clean), len(t2_clean))
        if max_len > 0:
            edit_dist = calculate_levenshtein_distance(t1_clean, t2_clean)
            edit_sim = 1.0 - (edit_dist / max_len)
        else:
            edit_sim = 1.0
    else:
        edit_sim = 1.0
    
    # Combine similarities (weighted average)
    # Jaccard is more important for word-based content
    # Edit distance catches small OCR errors
    combined = 0.6 * jaccard_sim + 0.4 * edit_sim
    
    return combined


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
    # Remove punctuation
    t1 = re.sub(r'[^\w\s]', '', text1.lower())
    t2 = re.sub(r'[^\w\s]', '', text2.lower())
    
    words1 = set(t1.split())
    words2 = set(t2.split())
    
    if not words1 and not words2:
        return 1.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def calculate_exact_match(text1: str, text2: str, strict: bool = False) -> bool:
    """
    Check if texts match exactly.
    
    Args:
        text1: First text
        text2: Second text
        strict: If False, normalize whitespace and case. If True, exact match.
    """
    if strict:
        return text1 == text2
    
    # Normalize
    t1 = ' '.join(text1.lower().split())
    t2 = ' '.join(text2.lower().split())
    # Remove punctuation
    t1 = re.sub(r'[^\w\s]', '', t1)
    t2 = re.sub(r'[^\w\s]', '', t2)
    
    return t1 == t2


def calculate_content_overlap(pred_text: str, gt_text: str) -> float:
    """
    Calculate if GT text is contained within pred text.
    Useful for merged cells that contain multiple GT cells.
    
    Returns score from 0 to 1.
    """
    if not gt_text:
        return 1.0 if not pred_text else 0.0
    if not pred_text:
        return 0.0
    
    pred_normalized = pred_text.lower().strip()
    gt_normalized = gt_text.lower().strip()
    
    # Check if GT is substring of prediction
    if gt_normalized in pred_normalized:
        return 1.0
    
    # Check word-level containment
    pred_words = set(pred_normalized.split())
    gt_words = set(gt_normalized.split())
    
    if gt_words and gt_words.issubset(pred_words):
        return 0.9
    
    # Partial word overlap
    if gt_words and pred_words:
        overlap = len(gt_words & pred_words)
        return overlap / len(gt_words)
    
    return 0.0


def match_cells(
    pred_cells: List[Cell],
    gt_cells: List[Cell],
    iou_threshold: float = 0.5,
    text_similarity_threshold: float = 0.5,
    use_text_fallback: bool = True,
    use_any_overlap: bool = True
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Match predicted cells with ground truth cells.
    
    Uses multi-stage matching:
    1. IoU-based matching for cells with good spatial overlap
    2. Any spatial overlap (even partial) with good content match
    3. Text-based fallback for unmatched cells with similar content
    
    Args:
        pred_cells: Predicted cells
        gt_cells: Ground truth cells
        iou_threshold: Minimum IoU for spatial matching
        text_similarity_threshold: Minimum text similarity for fallback matching
        use_text_fallback: Whether to use text-based fallback matching
        use_any_overlap: Whether to match cells with any spatial overlap + good content
    
    Returns:
        - matches: List of (pred_idx, gt_idx, match_score) tuples
        - unmatched_pred: Indices of unmatched predicted cells
        - unmatched_gt: Indices of unmatched ground truth cells
    """
    matches = []
    matched_pred = set()
    matched_gt = set()
    
    # Stage 1: IoU-based matching (strict spatial)
    iou_scores = []
    for i, pred_cell in enumerate(pred_cells):
        for j, gt_cell in enumerate(gt_cells):
            iou = calculate_grid_iou(pred_cell, gt_cell)
            if iou >= iou_threshold:
                iou_scores.append((iou, i, j))
    
    # Sort by IoU descending
    iou_scores.sort(reverse=True, key=lambda x: x[0])
    
    # Greedy matching based on IoU
    for iou, pred_idx, gt_idx in iou_scores:
        if pred_idx not in matched_pred and gt_idx not in matched_gt:
            matches.append((pred_idx, gt_idx, iou))
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
    
    # Stage 2: Any overlap + content matching (for merged cells)
    if use_any_overlap:
        unmatched_pred_indices = [i for i in range(len(pred_cells)) if i not in matched_pred]
        unmatched_gt_indices = [i for i in range(len(gt_cells)) if i not in matched_gt]
        
        overlap_scores = []
        for pred_idx in unmatched_pred_indices:
            pred_cell = pred_cells[pred_idx]
            pred_text = pred_cell.normalized_text()
            
            for gt_idx in unmatched_gt_indices:
                gt_cell = gt_cells[gt_idx]
                gt_text = gt_cell.normalized_text()
                
                # Check for any spatial overlap
                iou = calculate_grid_iou(pred_cell, gt_cell)
                if iou > 0:  # Any overlap at all
                    # Check content overlap
                    content_score = calculate_content_overlap(pred_text, gt_text)
                    if content_score >= text_similarity_threshold:
                        # Combined score (spatial + content)
                        combined_score = 0.3 * iou + 0.7 * content_score
                        overlap_scores.append((combined_score, pred_idx, gt_idx))
        
        # Sort by combined score
        overlap_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Greedy matching
        for score, pred_idx, gt_idx in overlap_scores:
            if pred_idx not in matched_pred and gt_idx not in matched_gt:
                matches.append((pred_idx, gt_idx, score))
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)
    
    # Stage 3: Text-based fallback matching for unmatched cells (no spatial requirement)
    if use_text_fallback:
        unmatched_pred_indices = [i for i in range(len(pred_cells)) if i not in matched_pred]
        unmatched_gt_indices = [i for i in range(len(gt_cells)) if i not in matched_gt]
        
        text_scores = []
        for pred_idx in unmatched_pred_indices:
            pred_text = pred_cells[pred_idx].normalized_text()
            if not pred_text:  # Skip empty cells
                continue
                
            for gt_idx in unmatched_gt_indices:
                gt_text = gt_cells[gt_idx].normalized_text()
                if not gt_text:  # Skip empty cells
                    continue
                
                # Calculate fuzzy text similarity
                similarity = calculate_text_similarity_fuzzy(pred_text, gt_text)
                
                if similarity >= text_similarity_threshold:
                    text_scores.append((similarity, pred_idx, gt_idx))
        
        # Sort by similarity descending
        text_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Greedy matching based on text similarity
        for similarity, pred_idx, gt_idx in text_scores:
            if pred_idx not in matched_pred and gt_idx not in matched_gt:
                matches.append((pred_idx, gt_idx, similarity))
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)
    
    unmatched_pred = [i for i in range(len(pred_cells)) if i not in matched_pred]
    unmatched_gt = [i for i in range(len(gt_cells)) if i not in matched_gt]
    
    return matches, unmatched_pred, unmatched_gt


def evaluate_extraction(
    pred_cells: List[Cell],
    gt_cells: List[Cell],
    iou_threshold: float = 0.5,
    text_similarity_threshold: float = 0.5,
    use_text_fallback: bool = True,
    use_fuzzy_text: bool = True,
    use_any_overlap: bool = True
) -> Dict[str, Any]:
    """
    Evaluate extraction quality with multiple metrics.
    
    Args:
        pred_cells: Predicted cells
        gt_cells: Ground truth cells
        iou_threshold: IoU threshold for spatial matching
        text_similarity_threshold: Threshold for text-based fallback
        use_text_fallback: Enable text-based fallback matching
        use_fuzzy_text: Use fuzzy text matching instead of exact Jaccard
        use_any_overlap: Enable partial overlap + content matching
    
    Returns:
        Dictionary with scores and statistics.
    """
    # Match cells
    matches, unmatched_pred, unmatched_gt = match_cells(
        pred_cells, gt_cells, iou_threshold, text_similarity_threshold, 
        use_text_fallback, use_any_overlap
    )
    
    # Cell-level metrics with partial credit
    # For merged cells, give partial credit
    tp = len(matches)  # True positives
    fp = len(unmatched_pred)  # False positives
    fn = len(unmatched_gt)  # False negatives
    
    # Calculate weighted TP based on match quality
    weighted_tp = sum(min(score, 1.0) for _, _, score in matches)
    
    precision = weighted_tp / (weighted_tp + fp) if (weighted_tp + fp) > 0 else 0.0
    recall = weighted_tp / (weighted_tp + fn) if (weighted_tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Content accuracy for matched cells
    text_similarities = []
    exact_matches = 0
    iou_matched = 0
    overlap_matched = 0
    text_matched = 0
    
    for pred_idx, gt_idx, match_score in matches:
        pred_text = pred_cells[pred_idx].text()
        gt_text = gt_cells[gt_idx].text()
        
        # Determine match type
        if match_score >= iou_threshold:
            iou_matched += 1
        elif match_score >= 0.1 and match_score < iou_threshold:
            overlap_matched += 1
        else:
            text_matched += 1
        
        # Calculate text similarity with multiple methods
        if use_fuzzy_text:
            similarity = calculate_text_similarity_fuzzy(pred_text, gt_text)
        else:
            similarity = calculate_text_similarity(pred_text, gt_text)
        
        # Also check content containment for merged cells
        containment = calculate_content_overlap(pred_text, gt_text)
        
        # Use the better of the two scores
        final_similarity = max(similarity, containment)
        text_similarities.append(final_similarity)
        
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
    
    # # Overall score (weighted combination)
    # # Prioritize content over structure for tolerant evaluation
    # overall_score = (
    #     0.3 * f1 +  # Cell detection
    #     0.5 * avg_text_similarity +  # Text accuracy is most important
    #     0.1 * row_accuracy +  # Structure is less important
    #     0.1 * col_accuracy
    # )
    
    # If only care about table layout, not content
    overall_score = (
        0.5 * f1 +           # Cell detection most important
        0.1 * avg_text_similarity +
        0.2 * row_accuracy +
        0.2 * col_accuracy
    )
    
    # # If only care about extracting the text correctly
    # overall_score = (
    #     0.2 * f1 +
    #     0.7 * avg_text_similarity +  # Content is everything
    #     0.05 * row_accuracy +
    #     0.05 * col_accuracy
    # )
        
    return {
        # Cell detection metrics
        'cell_detection': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'weighted_tp': weighted_tp,
            'iou_matched': iou_matched,
            'overlap_matched': overlap_matched,
            'text_matched': text_matched
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
            'text_threshold': text_similarity_threshold,
            'matched_pairs': len(matches),
            'unmatched_predicted': len(unmatched_pred),
            'unmatched_ground_truth': len(unmatched_gt),
            'use_text_fallback': use_text_fallback,
            'use_fuzzy_text': use_fuzzy_text
        }
    }


def print_evaluation_report(scores: Dict[str, Any], detailed: bool = False) -> None:
    """Print formatted evaluation report."""
    print("\n" + "="*70)
    print("TABLE EXTRACTION EVALUATION REPORT (IMPROVED)")
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
    if 'weighted_tp' in cd:
        print(f"  Weighted TP: {cd['weighted_tp']:.2f}")
    if 'iou_matched' in cd and 'text_matched' in cd:
        matches_str = f"  Matched by IoU: {cd['iou_matched']}"
        if 'overlap_matched' in cd and cd['overlap_matched'] > 0:
            matches_str += f", Overlap+Content: {cd['overlap_matched']}"
        matches_str += f", Text: {cd['text_matched']}"
        print(matches_str)
    
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
        print(f"  Text Similarity Threshold: {m['text_threshold']}")
        print(f"  Matched Pairs: {m['matched_pairs']}")
        print(f"  Unmatched Predicted: {m['unmatched_predicted']}")
        print(f"  Unmatched Ground Truth: {m['unmatched_ground_truth']}")
        print(f"  Text Fallback Enabled: {m['use_text_fallback']}")
        print(f"  Fuzzy Text Matching: {m['use_fuzzy_text']}")
    
    print("\n" + "="*70)


def compare_files(
    pred_file: Path,
    gt_file: Path,
    iou_threshold: float = 0.5,
    text_similarity_threshold: float = 0.5,
    use_text_fallback: bool = True,
    use_fuzzy_text: bool = True,
    use_any_overlap: bool = True,
    detailed: bool = False
) -> Dict[str, Any]:
    """Compare two JSON files and return evaluation scores."""
    # Load cells
    pred_cells = load_cells_from_json(pred_file)
    gt_cells = load_cells_from_json(gt_file)
    
    # Evaluate
    scores = evaluate_extraction(
        pred_cells, gt_cells,
        iou_threshold=iou_threshold,
        text_similarity_threshold=text_similarity_threshold,
        use_text_fallback=use_text_fallback,
        use_fuzzy_text=use_fuzzy_text,
        use_any_overlap=use_any_overlap
    )
    
    return scores


def batch_evaluate(
    pred_dir: Path,
    gt_dir: Path,
    iou_threshold: float = 0.5,
    text_similarity_threshold: float = 0.5,
    use_text_fallback: bool = True,
    use_fuzzy_text: bool = True,
    use_any_overlap: bool = True,
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
            scores = compare_files(
                pred_file, gt_file,
                iou_threshold=iou_threshold,
                text_similarity_threshold=text_similarity_threshold,
                use_text_fallback=use_text_fallback,
                use_fuzzy_text=use_fuzzy_text,
                use_any_overlap=use_any_overlap
            )
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
        'text_similarity_threshold': text_similarity_threshold,
        'use_text_fallback': use_text_fallback,
        'use_fuzzy_text': use_fuzzy_text,
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
    print("BATCH EVALUATION REPORT (IMPROVED)")
    print("="*70)
    
    print(f"\nFiles Evaluated: {aggregated['num_files']}")
    if aggregated['failed_files'] > 0:
        print(f"Failed: {aggregated['failed_files']}")
    print(f"IoU Threshold: {aggregated['iou_threshold']}")
    print(f"Text Similarity Threshold: {aggregated['text_similarity_threshold']}")
    print(f"Text Fallback Matching: {aggregated['use_text_fallback']}")
    print(f"Fuzzy Text Matching: {aggregated['use_fuzzy_text']}")
    
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
        description="Improved table extraction scoring with tolerant matching"
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
        '--text-threshold',
        type=float,
        default=0.5,
        help='Text similarity threshold for fallback matching (default: 0.5)'
    )
    parser.add_argument(
        '--no-text-fallback',
        action='store_true',
        help='Disable text-based fallback matching'
    )
    parser.add_argument(
        '--no-fuzzy-text',
        action='store_true',
        help='Use simple Jaccard instead of fuzzy text matching'
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
    parser.add_argument(
        '--mode',
        choices=['lenient', 'moderate', 'strict'],
        default='moderate',
        help='Preset matching mode (overrides individual thresholds)'
    )
    
    args = parser.parse_args()
    
    # Apply preset modes
    if args.mode == 'lenient':
        iou_threshold = 0.1
        text_threshold = 0.3
        use_text_fallback = True
        use_fuzzy_text = True
        use_any_overlap = True
    elif args.mode == 'moderate':
        iou_threshold = 0.3
        text_threshold = 0.4
        use_text_fallback = True
        use_fuzzy_text = True
        use_any_overlap = True
    else:  # strict
        iou_threshold = 0.5
        text_threshold = 0.7
        use_text_fallback = False
        use_fuzzy_text = False
        use_any_overlap = False
    
    # Override with explicit arguments if provided
    if args.iou_threshold != 0.5:  # User explicitly set it
        iou_threshold = args.iou_threshold
    if args.text_threshold != 0.5:  # User explicitly set it
        text_threshold = args.text_threshold
    
    use_text_fallback = not args.no_text_fallback and (args.mode != 'strict')
    use_fuzzy_text = not args.no_fuzzy_text and (args.mode != 'strict')
    use_any_overlap = args.mode != 'strict'
    
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
        print(f"Mode: {args.mode} (IoU: {iou_threshold}, Text: {text_threshold})")
        
        scores = compare_files(
            args.pred, args.gt,
            iou_threshold=iou_threshold,
            text_similarity_threshold=text_threshold,
            use_text_fallback=use_text_fallback,
            use_fuzzy_text=use_fuzzy_text,
            use_any_overlap=use_any_overlap,
            detailed=args.detailed
        )
        print_evaluation_report(scores, args.detailed)
        
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(scores, f, indent=2)
            print(f"\nScores saved to: {args.output}")
    
    # Batch directory comparison
    elif args.pred.is_dir() and args.gt.is_dir():
        print(f"Mode: {args.mode} (IoU: {iou_threshold}, Text: {text_threshold})")
        
        aggregated = batch_evaluate(
            args.pred, args.gt,
            iou_threshold=iou_threshold,
            text_similarity_threshold=text_threshold,
            use_text_fallback=use_text_fallback,
            use_fuzzy_text=use_fuzzy_text,
            use_any_overlap=use_any_overlap,
            output_file=args.output
        )
        
        if aggregated:
            print_batch_report(aggregated)
    
    else:
        print("Error: Both paths must be either files or directories")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
