#!/usr/bin/env python3
"""
Evaluate MLP regressor trained with OpenAI embeddings on the test set.

This script loads a model trained with OpenAI embeddings, generates embeddings
for the test set, and computes evaluation metrics.
"""

import argparse
import json
import sys
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def get_openai_embeddings(
    texts: list,
    model: str = "text-embedding-3-small",
    batch_size: int = 50,
    show_progress: bool = True,
    max_retries: int = 3,
    retry_delay: float = 0.5,
    max_chars_per_text: int = 6000
) -> np.ndarray:
    """
    Generate embeddings using OpenAI's embedding API.
    """
    import time
    from openai import OpenAI
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Either set it in your environment or create a .env file with OPENAI_API_KEY=your-key"
        )
    
    client = OpenAI(api_key=api_key)
    
    # Truncate long texts
    def truncate_text(text, max_chars):
        return text[:max_chars] if len(text) > max_chars else text
    
    processed_texts = [truncate_text(t, max_chars_per_text) for t in texts]
    truncated_count = sum(1 for orig, proc in zip(texts, processed_texts) if len(orig) > len(proc))
    
    if truncated_count > 0:
        print(f"  Truncated {truncated_count} texts to {max_chars_per_text} chars")
    
    all_embeddings = []
    num_batches = (len(processed_texts) + batch_size - 1) // batch_size
    
    if show_progress:
        batch_iterator = tqdm(range(0, len(processed_texts), batch_size), desc="OpenAI embeddings", total=num_batches)
    else:
        batch_iterator = range(0, len(processed_texts), batch_size)
    
    for i in batch_iterator:
        batch_texts = processed_texts[i:i + batch_size]
        batch_texts = [t if t.strip() else " " for t in batch_texts]
        
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=model,
                    input=batch_texts
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"Failed to get embeddings after {max_retries} attempts: {e}")
    
    return np.array(all_embeddings, dtype=np.float32)


def extract_structural_features(json_str: str) -> np.ndarray:
    """
    Extract hand-engineered structural features from table JSON.
    Must match the training script exactly!
    """
    try:
        data = json.loads(json_str)
    except:
        return np.zeros(12)
    
    if not isinstance(data, list) or len(data) == 0:
        return np.zeros(12)
    
    num_rows = len(data)
    num_cols = len(data[0]) if data[0] else 0
    num_cells = num_rows * num_cols
    
    all_cells = [cell for row in data for cell in row]
    empty_cells = sum(1 for cell in all_cells if not cell or not str(cell).strip())
    empty_ratio = empty_cells / num_cells if num_cells > 0 else 0
    
    text_lengths = [len(str(cell)) for cell in all_cells if cell]
    avg_text_len = np.mean(text_lengths) if text_lengths else 0
    std_text_len = np.std(text_lengths) if text_lengths else 0
    max_text_len = max(text_lengths) if text_lengths else 0
    
    col_counts = [len(row) for row in data]
    col_consistency = 1.0 if len(set(col_counts)) == 1 else 0.0
    
    aspect_ratio = num_cols / num_rows if num_rows > 0 else 0
    density = (num_cells - empty_cells) / num_cells if num_cells > 0 else 0
    
    features = np.array([
        num_rows,
        num_cols,
        num_cells,
        empty_ratio,
        avg_text_len,
        std_text_len,
        max_text_len,
        col_consistency,
        aspect_ratio,
        density,
        np.log1p(num_cells),
        np.log1p(avg_text_len)
    ])
    
    return features


def load_model(model_dir: Path, device: str = 'cpu'):
    """Load the trained model and its configuration."""
    # Import from training script (has correct model architectures)
    # The src/mlp_regressor.py has different implementations
    import importlib.util
    
    train_script_path = Path(__file__).parent / 'train_mlp_regressor.py'
    spec = importlib.util.spec_from_file_location("train_mlp_regressor", train_script_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    
    MLPRegressor = train_module.MLPRegressor
    ImprovedMLPRegressor = train_module.ImprovedMLPRegressor
    DeepMLPRegressor = train_module.DeepMLPRegressor
    AttentionMLPRegressor = train_module.AttentionMLPRegressor
    
    # Load embedding config
    embedding_config_path = model_dir / 'embedding_model_config.json'
    if not embedding_config_path.exists():
        embedding_config_path = model_dir / 'embeddings' / 'embedding_config.json'
    
    with open(embedding_config_path, 'r') as f:
        embedding_config = json.load(f)
    
    print(f"Embedding config: {embedding_config}")
    
    # Load model config
    config_path = model_dir / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load model
    model_path = model_dir / 'best_model.pt'
    if not model_path.exists():
        model_path = model_dir / 'final_model.pt'
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    hyperparams = checkpoint['hyperparameters']
    
    print(f"Model hyperparameters: {hyperparams}")
    
    # Determine model class from hyperparams or state dict
    use_attention = hyperparams.get('use_attention_model', False)
    use_deep = hyperparams.get('use_deep_model', False)
    use_improved = hyperparams.get('use_improved_model', False)
    
    input_dim = hyperparams['input_dim']
    
    if use_attention:
        model = AttentionMLPRegressor(
            input_dim=input_dim,
            hidden_dims=hyperparams.get('hidden_dims'),
            dropout_rate=hyperparams.get('dropout_rate', 0.2),
            use_residual=True,
            use_layer_norm=hyperparams.get('use_layer_norm', False),
            attention_heads=hyperparams.get('attention_heads', 8)
        )
        model_type = "AttentionMLPRegressor"
    elif use_deep:
        model = DeepMLPRegressor(
            input_dim=input_dim,
            hidden_dims=hyperparams.get('hidden_dims'),
            dropout_rate=hyperparams.get('dropout_rate', 0.2),
            use_residual=True,
            use_layer_norm=hyperparams.get('use_layer_norm', False)
        )
        model_type = "DeepMLPRegressor"
    elif use_improved:
        model = ImprovedMLPRegressor(
            input_dim=input_dim,
            hidden_dim1=hyperparams['hidden_dim1'],
            hidden_dim2=hyperparams['hidden_dim2'],
            dropout_rate=hyperparams.get('dropout_rate', 0.2),
            use_residual=True
        )
        model_type = "ImprovedMLPRegressor"
    else:
        model = MLPRegressor(
            input_dim=input_dim,
            hidden_dim1=hyperparams['hidden_dim1'],
            hidden_dim2=hyperparams['hidden_dim2'],
            dropout_rate=hyperparams.get('dropout_rate', 0.2)
        )
        model_type = "MLPRegressor"
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, embedding_config, config, model_type


def compute_metrics(predictions, ground_truth):
    """Compute evaluation metrics."""
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
    
    abs_diff = np.abs(pred_scores - gt_scores)
    mae = np.mean(abs_diff)
    rmse = np.sqrt(np.mean((pred_scores - gt_scores) ** 2))
    mse = np.mean((pred_scores - gt_scores) ** 2)
    median_ae = np.median(abs_diff)
    
    correlation = np.corrcoef(pred_scores, gt_scores)[0, 1]
    
    ss_res = np.sum((gt_scores - pred_scores) ** 2)
    ss_tot = np.sum((gt_scores - np.mean(gt_scores)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    epsilon = 1e-8
    mape = np.mean(np.abs((gt_scores - pred_scores) / (gt_scores + epsilon))) * 100
    
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
        'acc_1pct': float(acc_1pct),
        'acc_5pct': float(acc_5pct),
        'acc_10pct': float(acc_10pct),
        'acc_15pct': float(acc_15pct),
        'num_samples': len(ids)
    }


def analyze_predictions(predictions, ground_truth, top_k=20):
    """Analyze prediction quality."""
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    plt.title('OpenAI Embeddings Model: Predictions vs Ground Truth (Test Set)', fontsize=14, fontweight='bold')
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
        description="Evaluate OpenAI embeddings model on test set"
    )
    parser.add_argument(
        '--model-dir',
        type=Path,
        required=True,
        help='Directory containing trained model (e.g., experiments/openai_embeddings4)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output file for evaluation results (default: <model-dir>/eval/evaluation.json)'
    )
    parser.add_argument(
        '--plot-dir',
        type=Path,
        default=None,
        help='Directory to save plots (default: <model-dir>/eval/plots)'
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
    
    # Set default output paths based on model dir
    if args.output is None:
        args.output = args.model_dir / 'eval' / 'evaluation.json'
    if args.plot_dir is None:
        args.plot_dir = args.model_dir / 'eval' / 'plots'
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    
    model, embedding_config, config, model_type = load_model(args.model_dir, args.device)
    
    print(f"\n✓ Loaded model from {args.model_dir}")
    print(f"  Model type: {model_type}")
    print(f"  OpenAI model: {embedding_config.get('openai_model', embedding_config.get('model_name'))}")
    print(f"  Use hybrid features: {embedding_config.get('use_hybrid_features', False)}")
    print(f"  Total feature dimension: {embedding_config.get('total_dim')}")
    
    # Load scaler if using hybrid features
    use_hybrid = embedding_config.get('use_hybrid_features', False)
    scaler = None
    if use_hybrid:
        scaler_path = args.model_dir / 'embeddings' / 'feature_scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✓ Loaded feature scaler from {scaler_path}")
    
    # Load test dataset
    print("\n" + "="*70)
    print("LOADING TEST DATASET")
    print("="*70)
    
    print("Loading test dataset from Hugging Face...")
    dataset = load_dataset("rayhu/table-extraction-evaluation", split='test')
    
    texts = []
    sample_ids = []
    ground_truth = {}
    
    for sample in tqdm(dataset, desc="Processing test samples"):
        generated_json = json.dumps(sample['generated'])
        texts.append(generated_json)
        sample_ids.append(sample['id'])
        ground_truth[sample['id']] = sample['similarity_score']
    
    print(f"Loaded {len(texts)} test samples")
    
    # Generate embeddings
    print("\n" + "="*70)
    print("GENERATING OPENAI EMBEDDINGS")
    print("="*70)
    
    openai_model = embedding_config.get('openai_model') or embedding_config.get('model_name', '').replace('openai/', '')
    print(f"Using OpenAI model: {openai_model}")
    
    embeddings = get_openai_embeddings(
        texts,
        model=openai_model,
        batch_size=100,
        show_progress=True
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Prepare features
    if use_hybrid:
        print("\nExtracting structural features...")
        struct_features = np.array([extract_structural_features(text) for text in tqdm(texts, desc="Structural features")])
        
        # Normalize structural features using the saved scaler
        if scaler is not None:
            struct_features = scaler.transform(struct_features)
        
        # Combine embeddings with structural features
        features = np.concatenate([embeddings, struct_features], axis=1)
        print(f"Combined features shape: {features.shape}")
    else:
        features = embeddings
    
    # Generate predictions
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS")
    print("="*70)
    
    features_tensor = torch.tensor(features, dtype=torch.float32).to(args.device)
    
    predictions = {}
    with torch.no_grad():
        scores = model(features_tensor).cpu().numpy()
    
    for sample_id, score in zip(sample_ids, scores):
        predictions[sample_id] = float(score)
    
    print(f"Generated {len(predictions)} predictions")
    
    # Compute metrics
    print("\n" + "="*70)
    print("COMPUTING METRICS")
    print("="*70)
    
    metrics = compute_metrics(predictions, ground_truth)
    analysis = analyze_predictions(predictions, ground_truth)
    
    # Prepare results
    results = {
        'model_dir': str(args.model_dir),
        'model_type': model_type,
        'embedding_model': openai_model,
        'use_hybrid_features': use_hybrid,
        'test_set_size': len(ground_truth),
        'num_predictions': len(predictions),
        'metrics': metrics,
        'analysis': analysis
    }
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {args.output}")
    
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

