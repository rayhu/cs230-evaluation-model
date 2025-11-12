#!/usr/bin/env python3
"""
Compare multiple trained MLP models.

This script loads training histories from multiple model directories
and creates comparative visualizations and statistics.
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_model_info(model_dir: Path):
    """Load model configuration and training history."""
    config_path = model_dir / 'config.json'
    history_path = model_dir / 'training_history.json'
    
    if not config_path.exists() or not history_path.exists():
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Get best validation metrics
    val_metrics = history['val_metrics']
    best_epoch = np.argmin([m['mae'] for m in val_metrics])
    best_metrics = val_metrics[best_epoch]
    
    return {
        'name': model_dir.name,
        'path': str(model_dir),
        'config': config,
        'history': history,
        'best_epoch': best_epoch + 1,
        'best_val_mae': best_metrics['mae'],
        'best_val_rmse': best_metrics['rmse'],
        'final_val_mae': val_metrics[-1]['mae'],
        'final_train_loss': history['train_loss'][-1],
        'epochs_trained': len(history['train_loss'])
    }


def create_comparison_table(models):
    """Print comparison table."""
    print("\n" + "="*120)
    print("MODEL COMPARISON")
    print("="*120)
    print()
    
    # Header
    print(f"{'Model':<35} {'Embedding':<25} {'Best MAE':<12} {'Best RMSE':<12} {'Epochs':<10} {'Dropout':<10}")
    print("-"*120)
    
    # Sort by best MAE
    sorted_models = sorted(models, key=lambda x: x['best_val_mae'])
    
    for model in sorted_models:
        config = model['config']
        embedding = config.get('model_name', 'N/A').replace('sentence-transformers/', '')
        
        print(f"{model['name']:<35} {embedding:<25} {model['best_val_mae']:<12.6f} "
              f"{model['best_val_rmse']:<12.6f} {model['epochs_trained']:<10} "
              f"{config.get('dropout', 0.0):<10.2f}")
    
    print()
    
    # Best model
    best = sorted_models[0]
    print(f"🏆 Best Model: {best['name']}")
    print(f"   MAE: {best['best_val_mae']:.6f} (epoch {best['best_epoch']})")
    print(f"   RMSE: {best['best_val_rmse']:.6f}")
    print()


def plot_comparison(models, output_dir: Path):
    """Create comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Training curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for model in models:
        history = model['history']
        epochs = range(1, len(history['train_loss']) + 1)
        val_mae = [m['mae'] for m in history['val_metrics']]
        
        # Plot MAE
        axes[0].plot(epochs, val_mae, marker='o', label=model['name'], linewidth=2, markersize=3)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation MAE', fontsize=12)
    axes[0].set_title('Validation MAE Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Best MAE comparison (bar chart)
    names = [m['name'][:25] for m in models]  # Truncate long names
    best_maes = [m['best_val_mae'] for m in models]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(models)))
    
    bars = axes[1].bar(range(len(models)), best_maes, color=colors)
    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].set_ylabel('Best Validation MAE', fontsize=12)
    axes[1].set_title('Best MAE Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mae) in enumerate(zip(bars, best_maes)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{mae:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = output_dir / 'model_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Comparison plot saved to: {plot_path}")
    plt.close()
    
    # Plot 3: Detailed metrics comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models))
    width = 0.35
    
    maes = [m['best_val_mae'] for m in models]
    rmses = [m['best_val_rmse'] for m in models]
    
    ax.bar(x - width/2, maes, width, label='MAE', alpha=0.8)
    ax.bar(x + width/2, rmses, width, label='RMSE', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('MAE and RMSE Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    metrics_plot_path = output_dir / 'metrics_comparison.png'
    plt.savefig(metrics_plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Metrics comparison saved to: {metrics_plot_path}")
    plt.close()


def save_comparison_json(models, output_path: Path):
    """Save comparison data as JSON."""
    comparison = {
        'models': [
            {
                'name': m['name'],
                'path': m['path'],
                'best_val_mae': m['best_val_mae'],
                'best_val_rmse': m['best_val_rmse'],
                'best_epoch': m['best_epoch'],
                'epochs_trained': m['epochs_trained'],
                'config': {
                    'model_name': m['config'].get('model_name'),
                    'hidden_dim1': m['config'].get('hidden_dim1'),
                    'hidden_dim2': m['config'].get('hidden_dim2'),
                    'dropout': m['config'].get('dropout'),
                    'learning_rate': m['config'].get('learning_rate'),
                    'batch_size': m['config'].get('batch_size')
                }
            }
            for m in models
        ]
    }
    
    # Add rankings
    sorted_models = sorted(models, key=lambda x: x['best_val_mae'])
    comparison['rankings'] = {
        'by_mae': [m['name'] for m in sorted_models],
        'best_model': sorted_models[0]['name'],
        'best_mae': sorted_models[0]['best_val_mae']
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"💾 Comparison data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple trained MLP models"
    )
    parser.add_argument(
        '--model-dirs',
        type=str,
        nargs='+',
        required=True,
        help='Paths to model directories to compare'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('experiments/comparison'),
        help='Output directory for comparison results'
    )
    parser.add_argument(
        '--output-json',
        type=Path,
        default=None,
        help='Output JSON file for comparison data'
    )
    
    args = parser.parse_args()
    
    # Load all models
    models = []
    print("\nLoading models...")
    for dir_str in args.model_dirs:
        model_dir = Path(dir_str)
        if not model_dir.exists():
            print(f"⚠️  Warning: {model_dir} does not exist, skipping")
            continue
        
        model_info = load_model_info(model_dir)
        if model_info is None:
            print(f"⚠️  Warning: Could not load info from {model_dir}, skipping")
            continue
        
        models.append(model_info)
        print(f"✓ Loaded: {model_info['name']}")
    
    if len(models) < 2:
        print("\n❌ Need at least 2 models to compare")
        return 1
    
    # Create comparison
    create_comparison_table(models)
    plot_comparison(models, args.output_dir)
    
    # Save JSON
    output_json = args.output_json or (args.output_dir / 'comparison.json')
    save_comparison_json(models, output_json)
    
    print("\n" + "="*120)
    print("✅ Comparison complete!")
    print("="*120)
    print()
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

