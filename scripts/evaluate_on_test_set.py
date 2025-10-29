#!/usr/bin/env python3
"""
Evaluate MLP regressor predictions against ground truth from Hugging Face dataset.

This script loads the test split from rayhu/table-extraction-evaluation,
extracts ground truth similarity scores, and compares them with model predictions.

Supports both TF-IDF (legacy) and Word2Vec models.
"""

import argparse
import json
import sys
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from gensim.models import Word2Vec

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_test_ground_truth():
    """
    Load test set ground truth from Hugging Face dataset.
    
    Returns:
        Dictionary mapping sample IDs to ground truth scores
    """
    print("Loading test dataset from Hugging Face...")
    dataset = load_dataset("rayhu/table-extraction-evaluation", split='test')
    
    ground_truth = {}
    for sample in tqdm(dataset, desc="Extracting ground truth"):
        # Use the ID as the key
        sample_id = sample['id']
        ground_truth[sample_id] = sample['similarity_score']
    
    print(f"Loaded {len(ground_truth)} ground truth scores")
    return ground_truth


def tokenize_json(text: str):
    """Tokenize JSON text into words."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def doc_to_vec(tokens, w2v_model):
    """Convert document tokens to average Word2Vec vector."""
    vectors = []
    for word in tokens:
        if word in w2v_model.wv:
            vectors.append(w2v_model.wv[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(w2v_model.vector_size)


def predict_all_test_samples(model, feature_extractor, scaler, device='cpu', use_word2vec=True):
    """
    Predict quality scores for all test samples.
    
    Args:
        model: Trained MLP model
        feature_extractor: Word2Vec model or TF-IDF vectorizer
        scaler: StandardScaler for feature normalization (Word2Vec only)
        device: Device to run inference on
        use_word2vec: Whether using Word2Vec (True) or TF-IDF (False)
    
    Returns:
        Dictionary mapping sample IDs to predicted scores
    """
    print("Loading test dataset for prediction...")
    dataset = load_dataset("rayhu/table-extraction-evaluation", split='test')
    
    predictions = {}
    
    print(f"Generating predictions using {'Word2Vec' if use_word2vec else 'TF-IDF'}...")
    for sample in tqdm(dataset, desc="Predicting"):
        sample_id = sample['id']
        
        # Convert generated table to JSON string
        text = json.dumps(sample['generated'])
        
        # Extract features
        if use_word2vec:
            # Word2Vec: tokenize, convert to vector, scale
            tokens = tokenize_json(text)
            vec = doc_to_vec(tokens, feature_extractor)
            features = scaler.transform([vec])
        else:
            # TF-IDF (legacy)
            features = feature_extractor.transform([text]).toarray()
        
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        
        # Predict
        with torch.no_grad():
            score = model(features_tensor).item()
        
        predictions[sample_id] = score
    
    print(f"Generated {len(predictions)} predictions")
    return predictions


def compute_metrics(predictions, ground_truth):
    """Compute evaluation metrics."""
    # Align predictions and ground truth
    ids = []
    pred_scores = []
    gt_scores = []
    
    for sample_id in predictions:
        if sample_id in ground_truth:
            ids.append(sample_id)
            pred_scores.append(predictions[sample_id])
            gt_scores.append(ground_truth[sample_id])
    
    pred_scores = np.array(pred_scores)
    gt_scores = np.array(gt_scores)
    
    # Error metrics
    mae = np.mean(np.abs(pred_scores - gt_scores))
    rmse = np.sqrt(np.mean((pred_scores - gt_scores) ** 2))
    mse = np.mean((pred_scores - gt_scores) ** 2)
    
    # Correlation
    correlation = np.corrcoef(pred_scores, gt_scores)[0, 1]
    
    # R-squared
    ss_res = np.sum((gt_scores - pred_scores) ** 2)
    ss_tot = np.sum((gt_scores - np.mean(gt_scores)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Mean Absolute Percentage Error
    mask = gt_scores > 1e-6
    if np.any(mask):
        mape = np.mean(np.abs((gt_scores[mask] - pred_scores[mask]) / gt_scores[mask])) * 100
    else:
        mape = float('inf')
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mse': float(mse),
        'correlation': float(correlation),
        'r2_score': float(r2),
        'mape': float(mape) if mape != float('inf') else None,
        'num_samples': len(ids)
    }


def analyze_predictions(predictions, ground_truth, top_k=20):
    """Analyze prediction quality."""
    # Align data
    ids = []
    pred_scores = []
    gt_scores = []
    
    for sample_id in predictions:
        if sample_id in ground_truth:
            ids.append(sample_id)
            pred_scores.append(predictions[sample_id])
            gt_scores.append(ground_truth[sample_id])
    
    pred_scores = np.array(pred_scores)
    gt_scores = np.array(gt_scores)
    errors = np.abs(pred_scores - gt_scores)
    
    # Worst predictions
    worst_indices = np.argsort(errors)[-top_k:][::-1]
    worst = [
        {
            'id': ids[i],
            'predicted': float(pred_scores[i]),
            'ground_truth': float(gt_scores[i]),
            'error': float(errors[i])
        }
        for i in worst_indices
    ]
    
    # Best predictions
    best_indices = np.argsort(errors)[:top_k]
    best = [
        {
            'id': ids[i],
            'predicted': float(pred_scores[i]),
            'ground_truth': float(gt_scores[i]),
            'error': float(errors[i])
        }
        for i in best_indices
    ]
    
    return {
        'worst_predictions': worst,
        'best_predictions': best,
        'statistics': {
            'predictions': {
                'mean': float(np.mean(pred_scores)),
                'std': float(np.std(pred_scores)),
                'min': float(np.min(pred_scores)),
                'max': float(np.max(pred_scores))
            },
            'ground_truth': {
                'mean': float(np.mean(gt_scores)),
                'std': float(np.std(gt_scores)),
                'min': float(np.min(gt_scores)),
                'max': float(np.max(gt_scores))
            },
            'errors': {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'min': float(np.min(errors)),
                'max': float(np.max(errors))
            }
        }
    }


def create_plots(predictions, ground_truth, output_dir):
    """Create visualization plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plots")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Align data
    pred_scores = []
    gt_scores = []
    
    for sample_id in predictions:
        if sample_id in ground_truth:
            pred_scores.append(predictions[sample_id])
            gt_scores.append(ground_truth[sample_id])
    
    pred_scores = np.array(pred_scores)
    gt_scores = np.array(gt_scores)
    
    # 1. Scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(gt_scores, pred_scores, alpha=0.5, s=20)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect prediction', linewidth=2)
    plt.xlabel('Ground Truth Score', fontsize=12)
    plt.ylabel('Predicted Score', fontsize=12)
    plt.title('MLP Predictions vs Ground Truth (Test Set)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_vs_ground_truth.png', dpi=300)
    plt.close()
    
    # 2. Error distribution
    errors = pred_scores - gt_scores
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Error (Predicted - Ground Truth)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
    plt.axvline(x=np.mean(errors), color='g', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(errors):.3f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=300)
    plt.close()
    
    # 3. Absolute error
    abs_errors = np.abs(errors)
    plt.figure(figsize=(10, 6))
    plt.hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('Absolute Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Absolute Errors', fontsize=14, fontweight='bold')
    plt.axvline(x=np.mean(abs_errors), color='r', linestyle='--', linewidth=2,
                label=f'MAE: {np.mean(abs_errors):.4f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'absolute_error_distribution.png', dpi=300)
    plt.close()
    
    print(f"\n📊 Plots saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MLP regressor on test set using Hugging Face dataset ground truth"
    )
    parser.add_argument(
        '--model-dir',
        type=Path,
        required=True,
        help='Directory containing trained model and vectorizer'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/test_evaluation.json'),
        help='Output file for evaluation results'
    )
    parser.add_argument(
        '--plot-dir',
        type=Path,
        default=Path('results/test_plots'),
        help='Directory to save plots'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device for inference'
    )
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        args.device = 'cpu'
    
    # Load model and feature extractor
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    
    import pickle
    from mlp_regressor import MLPRegressor
    
    # Detect model type (Word2Vec or TF-IDF)
    word2vec_path = args.model_dir / 'word2vec_model.bin'
    tfidf_path = args.model_dir / 'tfidf_vectorizer.pkl'
    
    if word2vec_path.exists():
        # Word2Vec model
        use_word2vec = True
        print("Detected Word2Vec model")
        
        # Load Word2Vec
        feature_extractor = Word2Vec.load(str(word2vec_path))
        print(f"✓ Loaded Word2Vec model from {word2vec_path}")
        print(f"  Vocabulary size: {len(feature_extractor.wv)}")
        print(f"  Vector dimension: {feature_extractor.vector_size}")
        
        # Load scaler
        scaler_path = args.model_dir / 'feature_scaler.pkl'
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✓ Loaded feature scaler from {scaler_path}")
        
    elif tfidf_path.exists():
        # TF-IDF model (legacy)
        use_word2vec = False
        print("Detected TF-IDF model (legacy)")
        
        with open(tfidf_path, 'rb') as f:
            feature_extractor = pickle.load(f)
        print(f"✓ Loaded TF-IDF vectorizer from {tfidf_path}")
        scaler = None
        
    else:
        raise FileNotFoundError(
            f"No feature extractor found in {args.model_dir}\n"
            f"Expected either:\n"
            f"  - {word2vec_path} (Word2Vec)\n"
            f"  - {tfidf_path} (TF-IDF)"
        )
    
    # Load model
    model_path = args.model_dir / 'best_model.pt'
    if not model_path.exists():
        model_path = args.model_dir / 'final_model.pt'
    
    checkpoint = torch.load(model_path, map_location=args.device)
    hyperparams = checkpoint['hyperparameters']
    
    model = MLPRegressor(
        input_dim=hyperparams['input_dim'],
        hidden_dim1=hyperparams['hidden_dim1'],
        hidden_dim2=hyperparams['hidden_dim2'],
        dropout_rate=hyperparams.get('dropout_rate', 0.0)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    print(f"✓ Loaded model from {model_path}")
    print(f"  Architecture: {hyperparams['input_dim']} -> {hyperparams['hidden_dim1']} -> {hyperparams['hidden_dim2']} -> 1")
    
    # Load ground truth
    print("\n" + "="*70)
    print("LOADING GROUND TRUTH")
    print("="*70)
    ground_truth = load_test_ground_truth()
    
    # Generate predictions
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS")
    print("="*70)
    predictions = predict_all_test_samples(model, feature_extractor, scaler, args.device, use_word2vec)
    
    # Compute metrics
    print("\n" + "="*70)
    print("COMPUTING METRICS")
    print("="*70)
    metrics = compute_metrics(predictions, ground_truth)
    
    # Analyze predictions
    analysis = analyze_predictions(predictions, ground_truth)
    
    # Prepare results
    results = {
        'model_dir': str(args.model_dir),
        'model_type': 'word2vec' if use_word2vec else 'tfidf',
        'test_set_size': len(ground_truth),
        'num_predictions': len(predictions),
        'metrics': metrics,
        'analysis': analysis
    }
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {args.output}")
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SET EVALUATION RESULTS")
    print("="*70)
    
    print(f"\n📊 Dataset:")
    print(f"  Test samples:     {len(ground_truth)}")
    print(f"  Predictions made: {metrics['num_samples']}")
    
    print(f"\n📈 Performance Metrics:")
    print(f"  MAE (Mean Absolute Error):      {metrics['mae']:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")
    print(f"  Correlation:                    {metrics['correlation']:.4f}")
    print(f"  R² Score:                       {metrics['r2_score']:.4f}")
    if metrics['mape'] is not None:
        print(f"  MAPE (Mean Abs % Error):        {metrics['mape']:.2f}%")
    
    print(f"\n📊 Prediction Statistics:")
    print(f"  Mean:   {analysis['statistics']['predictions']['mean']:.4f}")
    print(f"  Std:    {analysis['statistics']['predictions']['std']:.4f}")
    print(f"  Range:  [{analysis['statistics']['predictions']['min']:.4f}, "
          f"{analysis['statistics']['predictions']['max']:.4f}]")
    
    print(f"\n📉 Ground Truth Statistics:")
    print(f"  Mean:   {analysis['statistics']['ground_truth']['mean']:.4f}")
    print(f"  Std:    {analysis['statistics']['ground_truth']['std']:.4f}")
    print(f"  Range:  [{analysis['statistics']['ground_truth']['min']:.4f}, "
          f"{analysis['statistics']['ground_truth']['max']:.4f}]")
    
    print(f"\n❌ Top 5 Worst Predictions:")
    for i, pred in enumerate(analysis['worst_predictions'][:5], 1):
        print(f"  {i}. ID: {pred['id']}")
        print(f"     Predicted: {pred['predicted']:.4f}, "
              f"Actual: {pred['ground_truth']:.4f}, "
              f"Error: {pred['error']:.4f}")
    
    print(f"\n✅ Top 5 Best Predictions:")
    for i, pred in enumerate(analysis['best_predictions'][:5], 1):
        print(f"  {i}. ID: {pred['id']}")
        print(f"     Predicted: {pred['predicted']:.4f}, "
              f"Actual: {pred['ground_truth']:.4f}, "
              f"Error: {pred['error']:.4f}")
    
    # Generate plots
    if not args.no_plots:
        print(f"\n📊 Generating plots...")
        create_plots(predictions, ground_truth, args.plot_dir)
    
    print("\n" + "="*70)
    print("\n✅ Evaluation complete!")
    print(f"\nResults: {args.output}")
    if not args.no_plots:
        print(f"Plots:   {args.plot_dir}/")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

