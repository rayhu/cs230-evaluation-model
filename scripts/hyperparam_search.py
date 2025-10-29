#!/usr/bin/env python3
"""
Hyperparameter search for MLP regressor.

This script performs a grid search or random search over hyperparameter space
to find the optimal configuration for the table quality prediction model.
"""

import argparse
import json
import sys
from pathlib import Path
from itertools import product
import random
import subprocess
from datetime import datetime

import numpy as np


def grid_search_configs(search_space):
    """
    Generate all combinations for grid search.
    
    Args:
        search_space: Dict of hyperparameter names to lists of values
    
    Returns:
        List of hyperparameter configurations
    """
    keys = list(search_space.keys())
    values = list(search_space.values())
    
    configs = []
    for combination in product(*values):
        config = dict(zip(keys, combination))
        configs.append(config)
    
    return configs


def random_search_configs(search_space, n_samples=20):
    """
    Generate random combinations for random search.
    
    Args:
        search_space: Dict of hyperparameter names to lists of values
        n_samples: Number of random configurations to sample
    
    Returns:
        List of hyperparameter configurations
    """
    configs = []
    for _ in range(n_samples):
        config = {key: random.choice(values) for key, values in search_space.items()}
        configs.append(config)
    
    return configs


def run_training(config, output_dir, base_args):
    """
    Run training with given hyperparameters.
    
    Args:
        config: Hyperparameter configuration
        output_dir: Base output directory
        base_args: Additional base arguments (e.g., --limit)
    
    Returns:
        Validation MAE from training
    """
    # Create experiment name
    exp_name = "_".join([f"{k}{v}" for k, v in sorted(config.items())])
    exp_dir = output_dir / exp_name
    
    # Build command
    cmd = [
        sys.executable,
        "scripts/train_mlp_regressor.py",
        "--output-dir", str(exp_dir)
    ]
    
    # Add hyperparameters
    for key, value in config.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    # Add base arguments
    cmd.extend(base_args)
    
    print(f"\n{'='*70}")
    print(f"Training: {exp_name}")
    print(f"{'='*70}")
    print(f"Config: {config}")
    
    # Run training
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Extract validation MAE from training history
        history_file = exp_dir / "training_history.json"
        with open(history_file) as f:
            history = json.load(f)
        
        # Get best validation MAE
        val_maes = [epoch['mae'] for epoch in history['val_metrics']]
        best_mae = min(val_maes)
        
        print(f"✅ Best Validation MAE: {best_mae:.4f}")
        
        return best_mae
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        print(f"Error output: {e.stderr}")
        return float('inf')
    except Exception as e:
        print(f"❌ Error: {e}")
        return float('inf')


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for MLP regressor"
    )
    parser.add_argument(
        '--search-type',
        type=str,
        default='grid',
        choices=['grid', 'random', 'recommended'],
        help='Search strategy (default: grid)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=20,
        help='Number of random samples (for random search)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('experiments/hyperparam_search'),
        help='Base output directory for experiments'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit training samples (for fast testing)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs for each trial (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Base arguments for all runs
    base_args = ['--epochs', str(args.epochs)]
    if args.limit:
        base_args.extend(['--limit', str(args.limit)])
    
    # Define search space
    if args.search_type == 'recommended':
        # Test 3 recommended configurations
        configs = [
            {
                'batch_size': 64,
                'learning_rate': 0.0005,
                'hidden_dim1': 512,
                'hidden_dim2': 128,
                'dropout': 0.3,
                'max_features': 15000
            },
            {
                'batch_size': 128,
                'learning_rate': 0.001,
                'hidden_dim1': 256,
                'hidden_dim2': 64,
                'dropout': 0.2,
                'max_features': 10000
            },
            {
                'batch_size': 32,
                'learning_rate': 0.0003,
                'hidden_dim1': 1024,
                'hidden_dim2': 512,
                'dropout': 0.4,
                'max_features': 20000
            }
        ]
        print(f"\n🎯 Testing 3 recommended configurations")
        
    elif args.search_type == 'grid':
        # Grid search space (small for speed)
        search_space = {
            'batch_size': [32, 64, 128],
            'learning_rate': [0.0003, 0.0005, 0.001],
            'hidden_dim1': [256, 512],
            'hidden_dim2': [64, 128],
            'dropout': [0.2, 0.3],
            'max_features': [10000, 15000]
        }
        configs = grid_search_configs(search_space)
        print(f"\n🔍 Grid search: {len(configs)} configurations")
        
    else:  # random search
        # Random search space
        search_space = {
            'batch_size': [16, 32, 64, 128, 256],
            'learning_rate': [0.0001, 0.0003, 0.0005, 0.001, 0.003],
            'hidden_dim1': [128, 256, 512, 1024],
            'hidden_dim2': [32, 64, 128, 256, 512],
            'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'max_features': [5000, 10000, 15000, 20000]
        }
        configs = random_search_configs(search_space, args.n_samples)
        print(f"\n🎲 Random search: {args.n_samples} configurations")
    
    # Run experiments
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n\n{'#'*70}")
        print(f"Experiment {i}/{len(configs)}")
        print(f"{'#'*70}")
        
        mae = run_training(config, args.output_dir, base_args)
        
        results.append({
            'config': config,
            'val_mae': mae,
            'experiment_id': i
        })
    
    # Sort by validation MAE
    results.sort(key=lambda x: x['val_mae'])
    
    # Save results
    results_file = args.output_dir / f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n\n{'='*70}")
    print("HYPERPARAMETER SEARCH RESULTS")
    print(f"{'='*70}")
    print(f"\nTotal experiments: {len(results)}")
    print(f"Results saved to: {results_file}")
    
    print(f"\n🏆 TOP 5 CONFIGURATIONS:\n")
    
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. Validation MAE: {result['val_mae']:.4f}")
        print(f"   Config: {result['config']}")
        print()
    
    # Print best configuration as command
    best = results[0]
    print(f"\n🎯 BEST CONFIGURATION (MAE: {best['val_mae']:.4f}):\n")
    
    cmd_parts = ["python scripts/train_mlp_regressor.py"]
    for key, value in sorted(best['config'].items()):
        cmd_parts.append(f"  --{key.replace('_', '-')} {value}")
    cmd_parts.append("  --epochs 30")
    cmd_parts.append("  --output-dir experiments/mlp_best")
    
    print("```bash")
    print(" \\\n".join(cmd_parts))
    print("```")
    
    print(f"\n{'='*70}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

