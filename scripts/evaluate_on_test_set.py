#!/usr/bin/env python3
"""
Evaluate MLP regressor predictions against ground truth from Hugging Face dataset.

This script loads the test split from rayhu/table-extraction-evaluation,
extracts ground truth similarity scores, and compares them with model predictions.

Supports both TF-IDF (legacy), Word2Vec (legacy), and Sentence Transformer models.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

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


def predict_all_test_samples(model, feature_extractor, scaler, device='cpu', model_type='sentence_transformer', use_hybrid=False):
    """
    Predict quality scores for all test samples.
    
    Args:
        model: Trained MLP model
        feature_extractor: Sentence Transformer model, Word2Vec model, or TF-IDF vectorizer
        scaler: StandardScaler for feature normalization
        device: Device to run inference on
        model_type: Type of feature extractor ('sentence_transformer', 'word2vec', or 'tfidf')
        use_hybrid: Whether to use hybrid features (structure + text + embeddings)
    
    Returns:
        Dictionary mapping sample IDs to predicted scores
    """
    print("Loading test dataset for prediction...")
    dataset = load_dataset("rayhu/table-extraction-evaluation", split='test')
    
    # Convert to texts
    texts = [json.dumps(sample['generated']) for sample in dataset]
    sample_ids = [sample['id'] for sample in dataset]
    
    print(f"Generating predictions using {model_type}...")
    if use_hybrid:
        print("  Using hybrid features (structure + text stats + embeddings)")
    
    if model_type == 'sentence_transformer':
        if use_hybrid:
            # Extract hybrid features
            from utils.table_features import extract_all_features
            features = []
            for text in tqdm(texts, desc="Extracting hybrid features"):
                feat = extract_all_features(
                    text,
                    sentence_transformer=feature_extractor,
                    normalize_embeddings=False
                )
                features.append(feat)
            features = np.array(features)
        else:
            # Encode all texts at once with sentence transformer
            features = feature_extractor.encode(
                texts,
                show_progress_bar=True,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=False
            )
        # Standardize
        features = scaler.transform(features)
        
    elif model_type == 'word2vec':
        # Legacy Word2Vec support
        from gensim.models import Word2Vec
        import re
        from collections import Counter
        
        def tokenize_json(text: str):
            tokens = re.findall(r'\b\w+\b', text.lower())
            return tokens
        
        def doc_to_vec(tokens, w2v_model, idf_scores=None):
            if idf_scores is None:
                vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
                return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)
            else:
                vectors, weights = [], []
                token_counts = Counter(tokens)
                total_tokens = len(tokens)
                for word in set(tokens):
                    if word in w2v_model.wv:
                        tf = token_counts[word] / total_tokens
                        idf_score = idf_scores.get(word, 1.0)
                        vectors.append(w2v_model.wv[word])
                        weights.append(tf * idf_score)
                if vectors:
                    return np.average(np.array(vectors), axis=0, weights=np.array(weights))
                return np.zeros(w2v_model.vector_size)
        
        features = []
        for text in tqdm(texts, desc="Extracting Word2Vec features"):
            tokens = tokenize_json(text)
            vec = doc_to_vec(tokens, feature_extractor, None)
            features.append(vec)
        features = np.array(features)
        features = scaler.transform(features)
        
    else:  # tfidf
        # Legacy TF-IDF support
        features = feature_extractor.transform(texts).toarray()
    
    # Batch prediction
    predictions = {}
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        scores = model(features_tensor).cpu().numpy()
    
    for sample_id, score in zip(sample_ids, scores):
        predictions[sample_id] = float(score)
    
    print(f"Generated {len(predictions)} predictions")
    return predictions


def compute_metrics(predictions, ground_truth):
    """
    Compute comprehensive evaluation metrics for percentage/score predictions.
    
    These metrics are more appropriate than exact matching for percentage comparisons
    because they account for the relative magnitude of errors and provide tolerance-based
    assessments that are more meaningful for continuous scores in [0, 1] range.
    """
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
    abs_diff = np.abs(pred_scores - gt_scores)
    mae = np.mean(abs_diff)
    rmse = np.sqrt(np.mean((pred_scores - gt_scores) ** 2))
    mse = np.mean((pred_scores - gt_scores) ** 2)
    
    # Median Absolute Error (more robust to outliers)
    median_ae = np.median(abs_diff)
    
    # Correlation
    correlation = np.corrcoef(pred_scores, gt_scores)[0, 1]
    
    # R-squared (Coefficient of Determination)
    ss_res = np.sum((gt_scores - pred_scores) ** 2)
    ss_tot = np.sum((gt_scores - np.mean(gt_scores)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Mean Absolute Percentage Error (normalized by ground truth)
    epsilon = 1e-8
    mape = np.mean(np.abs((gt_scores - pred_scores) / (gt_scores + epsilon))) * 100
    
    # Tolerance-based accuracy (percentage within ±1%, ±5%, ±10%, ±15%)
    acc_1pct = (abs_diff <= 0.01).astype(float).mean() * 100
    acc_5pct = (abs_diff <= 0.05).astype(float).mean() * 100
    acc_10pct = (abs_diff <= 0.10).astype(float).mean() * 100
    acc_15pct = (abs_diff <= 0.15).astype(float).mean() * 100
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mse': float(mse),
        'median_ae': float(median_ae),
        'correlation': float(correlation),
        'r2_score': float(r2),
        'mape': float(mape),
        'acc_1pct': float(acc_1pct),   # % predictions within ±0.01
        'acc_5pct': float(acc_5pct),   # % predictions within ±0.05
        'acc_10pct': float(acc_10pct), # % predictions within ±0.10
        'acc_15pct': float(acc_15pct), # % predictions within ±0.15
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
    from mlp_regressor import MLPRegressor, ImprovedMLPRegressor, DeepMLPRegressor
    
    # Detect model type (Sentence Transformer, Word2Vec, or TF-IDF)
    sentence_transformer_config_path = args.model_dir / 'sentence_transformer_config.json'
    word2vec_path = args.model_dir / 'word2vec_model.bin'
    tfidf_path = args.model_dir / 'tfidf_vectorizer.pkl'
    
    if sentence_transformer_config_path.exists():
        # Sentence Transformer model
        model_type = 'sentence_transformer'
        print("Detected Sentence Transformer model")
        
        # Load config
        with open(sentence_transformer_config_path, 'r') as f:
            config = json.load(f)
        model_name = config['model_name']
        use_hybrid_features = config.get('use_hybrid_features', False)
        
        # Load Sentence Transformer
        feature_extractor = SentenceTransformer(model_name)
        print(f"✓ Loaded Sentence Transformer: {model_name}")
        print(f"  Embedding dimension: {feature_extractor.get_sentence_embedding_dimension()}")
        if use_hybrid_features:
            print(f"  Using hybrid features (structure + text + embeddings)")
        
        # Load scaler
        scaler_path = args.model_dir / 'feature_scaler.pkl'
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✓ Loaded feature scaler from {scaler_path}")
        
    elif word2vec_path.exists():
        # Word2Vec model (legacy)
        from gensim.models import Word2Vec
        model_type = 'word2vec'
        use_hybrid_features = False
        print("Detected Word2Vec model (legacy)")
        
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
        model_type = 'tfidf'
        use_hybrid_features = False
        print("Detected TF-IDF model (legacy)")
        
        with open(tfidf_path, 'rb') as f:
            feature_extractor = pickle.load(f)
        print(f"✓ Loaded TF-IDF vectorizer from {tfidf_path}")
        scaler = None
        
    else:
        raise FileNotFoundError(
            f"No feature extractor found in {args.model_dir}\n"
            f"Expected either:\n"
            f"  - {sentence_transformer_config_path} (Sentence Transformer)\n"
            f"  - {word2vec_path} (Word2Vec)\n"
            f"  - {tfidf_path} (TF-IDF)"
        )
    
    # Load model
    model_path = args.model_dir / 'best_model.pt'
    if not model_path.exists():
        model_path = args.model_dir / 'final_model.pt'
    
    checkpoint = torch.load(model_path, map_location=args.device)
    hyperparams = checkpoint['hyperparameters']
    
    # Detect model class from state dict keys
    state_dict_keys = list(checkpoint['model_state_dict'].keys())
    
    if any('layers.' in key for key in state_dict_keys):
        # DeepMLPRegressor
        model_class = DeepMLPRegressor
        model_name = "DeepMLPRegressor"
        # Extract hidden dims from state dict if available
        hidden_dims = hyperparams.get('hidden_dims', None)
        model = DeepMLPRegressor(
            input_dim=hyperparams['input_dim'],
            hidden_dims=hidden_dims,
            dropout_rate=hyperparams.get('dropout_rate', 0.0),
            use_residual=hyperparams.get('use_residual', True),
            use_layer_norm=hyperparams.get('use_layer_norm', False)
        )
    elif any('fc1.' in key for key in state_dict_keys):
        # ImprovedMLPRegressor
        model_class = ImprovedMLPRegressor
        model_name = "ImprovedMLPRegressor"
        model = ImprovedMLPRegressor(
            input_dim=hyperparams['input_dim'],
            hidden_dim1=hyperparams['hidden_dim1'],
            hidden_dim2=hyperparams['hidden_dim2'],
            dropout_rate=hyperparams.get('dropout_rate', 0.0),
            use_residual=hyperparams.get('use_residual', True)
        )
    else:
        # Basic MLPRegressor
        model_class = MLPRegressor
        model_name = "MLPRegressor"
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
    print(f"  Model type: {model_name}")
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
    predictions = predict_all_test_samples(model, feature_extractor, scaler, args.device, model_type, use_hybrid_features)
    
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
        'model_type': model_type,
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
    print(f"  Median AE (Median Abs Error):   {metrics['median_ae']:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")
    print(f"  MAPE (Mean Abs % Error):        {metrics['mape']:.2f}%")
    print(f"  Correlation:                    {metrics['correlation']:.4f}")
    print(f"  R² Score:                       {metrics['r2_score']:.4f}")
    print(f"\n📊 Tolerance-based Accuracy:")
    print(f"  Within ±1%  (±0.01):            {metrics['acc_1pct']:.1f}%")
    print(f"  Within ±5%  (±0.05):            {metrics['acc_5pct']:.1f}%")
    print(f"  Within ±10% (±0.10):            {metrics['acc_10pct']:.1f}%")
    print(f"  Within ±15% (±0.15):            {metrics['acc_15pct']:.1f}%")
    
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

