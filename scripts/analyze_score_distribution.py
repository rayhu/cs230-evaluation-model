#!/usr/bin/env python3
"""
Analyze and visualize the score distribution in metadata JSONL files or Parquet datasets.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    # Fallback if numpy not available
    HAS_NUMPY = False
    class _NumpyFallback:  # type: ignore
        @staticmethod
        def std(x):
            mean = sum(x) / len(x)
            return (sum((v - mean) ** 2 for v in x) / len(x)) ** 0.5
        @staticmethod
        def median(x):
            sorted_x = sorted(x)
            n = len(sorted_x)
            if n % 2 == 0:
                return (sorted_x[n//2 - 1] + sorted_x[n//2]) / 2
            return sorted_x[n//2]
    np = _NumpyFallback()  # type: ignore

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def analyze_distribution_from_jsonl(metadata_file: Path) -> dict:
    """Analyze score distribution from JSONL file."""
    scores = []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            scores.append(data['similarity_score'])
    
    return _compute_statistics(scores)


def analyze_distribution_from_parquet(parquet_dir: Path, split: str = 'train') -> dict:
    """Analyze score distribution from Parquet dataset."""
    if not HAS_DATASETS:
        raise ImportError("datasets library is required for Parquet support. Install with: pip install datasets")
    
    print(f"Loading dataset from: {parquet_dir}")
    dataset = load_dataset(str(parquet_dir), split=split)
    print(f"Loaded {len(dataset)} samples")
    
    scores = [sample['similarity_score'] for sample in dataset]
    return _compute_statistics(scores)


def _compute_statistics(scores: list) -> dict:
    """Compute distribution statistics from a list of scores."""
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
        'std': np.std(scores),
        'min': min(scores),
        'max': max(scores),
        'median': np.median(scores),
        'distribution': distribution,
        'scores': scores
    }


def plot_distribution(analysis: dict, output_file: Optional[Path] = None):
    """Plot score distribution histogram."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plot generation")
        return
    
    scores = analysis['scores']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(scores, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(analysis['mean'], color='red', linestyle='--', label=f"Mean: {analysis['mean']:.3f}")
    ax1.axvline(analysis['median'], color='green', linestyle='--', label=f"Median: {analysis['median']:.3f}")
    ax1.set_xlabel('Similarity Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Score Distribution Histogram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bar chart by buckets
    labels = []
    counts = []
    for label, info in analysis['distribution'].items():
        labels.append(f"{label}\n({info['range'][0]:.1f}-{info['range'][1]:.1f})")
        counts.append(info['count'])
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    ax2.bar(labels, counts, color=colors, edgecolor='black', alpha=0.7)
    ax2.set_ylabel('Count')
    ax2.set_title('Score Distribution by Buckets')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    for i, (label, info) in enumerate(analysis['distribution'].items()):
        ax2.text(i, info['count'] + max(counts) * 0.01, 
                f"{info['percentage']:.1f}%", 
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze score distribution in metadata JSONL file or Parquet dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze JSONL file:
  python scripts/analyze_score_distribution.py --metadata dataset/train/metadata_train.jsonl --plot results/distribution.png
  
  # Analyze Parquet dataset:
  python scripts/analyze_score_distribution.py --parquet dataset_parquet_v3 --split train --plot results/distribution_v3.png
  
  # Analyze from Hugging Face:
  python scripts/analyze_score_distribution.py --parquet rayhu/table-extraction-evaluation --split train
        """
    )
    parser.add_argument(
        '--metadata',
        type=Path,
        help='Metadata JSONL file to analyze (mutually exclusive with --parquet)'
    )
    parser.add_argument(
        '--parquet',
        type=str,
        help='Parquet dataset directory or Hugging Face dataset ID (mutually exclusive with --metadata)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='Dataset split to analyze (default: train)'
    )
    parser.add_argument(
        '--plot',
        type=Path,
        help='Save distribution plot to this file (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.metadata and not args.parquet:
        print("Error: Either --metadata or --parquet must be provided")
        return 1
    
    if args.metadata and args.parquet:
        print("Error: --metadata and --parquet are mutually exclusive")
        return 1
    
    print("Analyzing score distribution...")
    
    # Analyze based on input type
    if args.metadata:
        if not args.metadata.exists():
            print(f"Error: Metadata file does not exist: {args.metadata}")
            return 1
        analysis = analyze_distribution_from_jsonl(args.metadata)
    else:
        # Parquet dataset
        parquet_path = Path(args.parquet) if Path(args.parquet).exists() else args.parquet
        try:
            analysis = analyze_distribution_from_parquet(parquet_path, args.split)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return 1
    
    if not analysis:
        print("Error: Could not analyze distribution")
        return 1
    
    print("\n" + "="*70)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("="*70)
    print(f"Total samples: {analysis['total']}")
    print(f"Mean score: {analysis['mean']:.3f}")
    print(f"Median score: {analysis['median']:.3f}")
    print(f"Std deviation: {analysis['std']:.3f}")
    print(f"Score range: {analysis['min']:.3f} - {analysis['max']:.3f}")
    
    print("\nScore Distribution by Buckets:")
    print("-" * 70)
    for label, info in analysis['distribution'].items():
        bar = "█" * int(info['percentage'] / 2)
        print(f"{label:12} ({info['range'][0]:.1f}-{info['range'][1]:.1f}): "
              f"{info['count']:5d} ({info['percentage']:5.1f}%) {bar}")
    
    if args.plot:
        plot_distribution(analysis, args.plot)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

