#!/usr/bin/env python3
"""
Compare predicted quality scores with ground truth scores.

This script analyzes how well the MLP regressor predicts actual quality scores
by computing correlation, MAE, RMSE, and other metrics.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_predictions(pred_file):
    """Load predictions from predict_quality.py output."""
    with open(pred_file) as f:
        data = json.load(f)
    
    # Extract predictions as dict {filename: score}
    pred_dict = {}
    for item in data['predictions']:
        pred_dict[item['filename']] = item['predicted_score']
    
    return pred_dict


def load_ground_truth(gt_file):
    """Load ground truth scores from score_extraction.py output."""
    with open(gt_file) as f:
        data = json.load(f)
    
    # Extract ground truth scores as dict {filename: score}
    gt_dict = {}
    for item in data['individual_scores']:
        gt_dict[item['filename']] = item['overall_score']
    
    return gt_dict


def align_data(pred_dict, gt_dict):
    """Align predictions and ground truth by filename."""
    filenames = []
    predictions = []
    ground_truth = []
    
    for filename in pred_dict:
        if filename in gt_dict:
            filenames.append(filename)
            predictions.append(pred_dict[filename])
            ground_truth.append(gt_dict[filename])
    
    return filenames, np.array(predictions), np.array(ground_truth)


def compute_metrics(predictions, ground_truth):
    """Compute evaluation metrics."""
    # Error metrics
    mae = np.mean(np.abs(predictions - ground_truth))
    rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
    mse = np.mean((predictions - ground_truth) ** 2)
    
    # Correlation
    correlation = np.corrcoef(predictions, ground_truth)[0, 1]
    
    # R-squared
    ss_res = np.sum((ground_truth - predictions) ** 2)
    ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Mean Absolute Percentage Error
    # Avoid division by zero
    mask = ground_truth > 1e-6
    if np.any(mask):
        mape = np.mean(np.abs((ground_truth[mask] - predictions[mask]) / ground_truth[mask])) * 100
    else:
        mape = float('inf')
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mse': mse,
        'correlation': correlation,
        'r2_score': r2,
        'mape': mape
    }


def analyze_errors(predictions, ground_truth, filenames, top_k=10):
    """Analyze prediction errors."""
    errors = np.abs(predictions - ground_truth)
    
    # Top errors
    top_error_indices = np.argsort(errors)[-top_k:][::-1]
    top_errors = [
        {
            'filename': filenames[i],
            'predicted': float(predictions[i]),
            'ground_truth': float(ground_truth[i]),
            'error': float(errors[i])
        }
        for i in top_error_indices
    ]
    
    # Best predictions
    best_indices = np.argsort(errors)[:top_k]
    best_predictions = [
        {
            'filename': filenames[i],
            'predicted': float(predictions[i]),
            'ground_truth': float(ground_truth[i]),
            'error': float(errors[i])
        }
        for i in best_indices
    ]
    
    return {
        'worst_predictions': top_errors,
        'best_predictions': best_predictions
    }


def plot_results(predictions, ground_truth, output_dir):
    """Create visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Scatter plot: Predictions vs Ground Truth
    plt.figure(figsize=(10, 8))
    plt.scatter(ground_truth, predictions, alpha=0.5, s=20)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
    plt.xlabel('Ground Truth Score', fontsize=12)
    plt.ylabel('Predicted Score', fontsize=12)
    plt.title('Predicted vs Ground Truth Quality Scores', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_vs_ground_truth.png', dpi=300)
    plt.close()
    
    # 2. Error distribution
    errors = predictions - ground_truth
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Error (Predicted - Ground Truth)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='r', linestyle='--', label='Zero error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=300)
    plt.close()
    
    # 3. Absolute error distribution
    abs_errors = np.abs(errors)
    plt.figure(figsize=(10, 6))
    plt.hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('Absolute Prediction Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Absolute Errors', fontsize=14, fontweight='bold')
    plt.axvline(x=np.mean(abs_errors), color='r', linestyle='--', 
                label=f'Mean: {np.mean(abs_errors):.3f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'absolute_error_distribution.png', dpi=300)
    plt.close()
    
    print(f"\n📊 Plots saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Compare predicted quality scores with ground truth"
    )
    parser.add_argument(
        '--predictions',
        type=Path,
        required=True,
        help='Predictions JSON file from predict_quality.py'
    )
    parser.add_argument(
        '--ground-truth',
        type=Path,
        required=True,
        help='Ground truth scores JSON from score_extraction.py'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/prediction_analysis.json'),
        help='Output file for analysis results'
    )
    parser.add_argument(
        '--plot-dir',
        type=Path,
        default=Path('results/plots'),
        help='Directory to save plots'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading predictions and ground truth...")
    pred_dict = load_predictions(args.predictions)
    gt_dict = load_ground_truth(args.ground_truth)
    
    print(f"Predictions: {len(pred_dict)} files")
    print(f"Ground truth: {len(gt_dict)} files")
    
    # Align data
    filenames, predictions, ground_truth = align_data(pred_dict, gt_dict)
    print(f"Matched: {len(filenames)} files")
    
    if len(filenames) == 0:
        print("Error: No matching files found!")
        return 1
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, ground_truth)
    
    # Analyze errors
    print("Analyzing errors...")
    error_analysis = analyze_errors(predictions, ground_truth, filenames)
    
    # Prepare results
    results = {
        'num_samples': len(filenames),
        'metrics': metrics,
        'error_analysis': error_analysis,
        'statistics': {
            'predictions': {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions))
            },
            'ground_truth': {
                'mean': float(np.mean(ground_truth)),
                'std': float(np.std(ground_truth)),
                'min': float(np.min(ground_truth)),
                'max': float(np.max(ground_truth))
            }
        }
    }
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    # Print summary
    print("\n" + "="*70)
    print("PREDICTION ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nSamples: {len(filenames)}")
    print(f"\n📊 Performance Metrics:")
    print(f"  MAE (Mean Absolute Error):      {metrics['mae']:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")
    print(f"  Correlation:                    {metrics['correlation']:.4f}")
    print(f"  R² Score:                       {metrics['r2_score']:.4f}")
    if metrics['mape'] != float('inf'):
        print(f"  MAPE (Mean Abs % Error):        {metrics['mape']:.2f}%")
    
    print(f"\n📈 Prediction Statistics:")
    print(f"  Mean:   {results['statistics']['predictions']['mean']:.4f}")
    print(f"  Std:    {results['statistics']['predictions']['std']:.4f}")
    print(f"  Range:  [{results['statistics']['predictions']['min']:.4f}, "
          f"{results['statistics']['predictions']['max']:.4f}]")
    
    print(f"\n📉 Ground Truth Statistics:")
    print(f"  Mean:   {results['statistics']['ground_truth']['mean']:.4f}")
    print(f"  Std:    {results['statistics']['ground_truth']['std']:.4f}")
    print(f"  Range:  [{results['statistics']['ground_truth']['min']:.4f}, "
          f"{results['statistics']['ground_truth']['max']:.4f}]")
    
    print(f"\n❌ Top 5 Worst Predictions:")
    for i, error in enumerate(error_analysis['worst_predictions'][:5], 1):
        print(f"  {i}. {error['filename']}")
        print(f"     Predicted: {error['predicted']:.4f}, "
              f"Actual: {error['ground_truth']:.4f}, "
              f"Error: {error['error']:.4f}")
    
    print(f"\n✅ Top 5 Best Predictions:")
    for i, pred in enumerate(error_analysis['best_predictions'][:5], 1):
        print(f"  {i}. {pred['filename']}")
        print(f"     Predicted: {pred['predicted']:.4f}, "
              f"Actual: {pred['ground_truth']:.4f}, "
              f"Error: {pred['error']:.4f}")
    
    # Generate plots
    if not args.no_plots:
        print(f"\nGenerating plots...")
        try:
            plot_results(predictions, ground_truth, args.plot_dir)
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
    
    print("\n" + "="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

