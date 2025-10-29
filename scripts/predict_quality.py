#!/usr/bin/env python3
"""
Predict table extraction quality using trained MLP regressor.

This script loads a trained model and TF-IDF vectorizer to predict quality
scores for new table extractions without requiring ground truth.
"""

import argparse
import json
import sys
from pathlib import Path
import pickle

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from mlp_regressor import MLPRegressor


def load_model_and_vectorizer(model_dir: Path):
    """
    Load trained model and TF-IDF vectorizer.
    
    Args:
        model_dir: Directory containing saved model and vectorizer
    
    Returns:
        model: Trained MLPRegressor
        vectorizer: Fitted TF-IDF vectorizer
    """
    # Load vectorizer
    vectorizer_path = model_dir / 'tfidf_vectorizer.pkl'
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load model
    model_path = model_dir / 'best_model.pt'
    if not model_path.exists():
        model_path = model_dir / 'final_model.pt'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found in {model_dir}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract hyperparameters
    hyperparams = checkpoint['hyperparameters']
    
    # Initialize model
    model = MLPRegressor(
        input_dim=hyperparams['input_dim'],
        hidden_dim1=hyperparams['hidden_dim1'],
        hidden_dim2=hyperparams['hidden_dim2'],
        dropout_rate=hyperparams.get('dropout_rate', 0.0)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from: {model_path}")
    print(f"Model architecture: {hyperparams['input_dim']} -> {hyperparams['hidden_dim1']} -> {hyperparams['hidden_dim2']} -> 1")
    
    return model, vectorizer


def predict_single(model, vectorizer, table_json, device='cpu'):
    """
    Predict quality score for a single table.
    
    Args:
        model: Trained model
        vectorizer: TF-IDF vectorizer
        table_json: Table structure as dictionary or JSON string
        device: Device to run inference on
    
    Returns:
        Predicted quality score (float)
    """
    # Convert to JSON string if needed
    if isinstance(table_json, dict):
        text = json.dumps(table_json)
    else:
        text = table_json
    
    # Extract features
    features = vectorizer.transform([text]).toarray()
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    
    # Predict
    with torch.no_grad():
        score = model(features_tensor).item()
    
    return score


def predict_batch(model, vectorizer, table_jsons, device='cpu'):
    """
    Predict quality scores for multiple tables.
    
    Args:
        model: Trained model
        vectorizer: TF-IDF vectorizer
        table_jsons: List of table structures (dicts or JSON strings)
        device: Device to run inference on
    
    Returns:
        List of predicted quality scores
    """
    # Convert to JSON strings
    texts = []
    for table_json in table_jsons:
        if isinstance(table_json, dict):
            texts.append(json.dumps(table_json))
        else:
            texts.append(table_json)
    
    # Extract features
    features = vectorizer.transform(texts).toarray()
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    
    # Predict
    with torch.no_grad():
        scores = model(features_tensor).cpu().numpy()
    
    return scores.tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Predict table extraction quality scores"
    )
    parser.add_argument(
        '--model-dir',
        type=Path,
        default=Path('experiments/mlp_regressor'),
        help='Directory containing trained model and vectorizer'
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input JSON file (single table) or directory (batch prediction)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file to save predictions (JSON format)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to run inference on (default: cpu)'
    )
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load model and vectorizer
    print("Loading model and vectorizer...")
    model, vectorizer = load_model_and_vectorizer(args.model_dir)
    model.to(args.device)
    
    # Single file prediction
    if args.input.is_file():
        print(f"\nPredicting quality for: {args.input}")
        
        with open(args.input, 'r') as f:
            table_data = json.load(f)
        
        score = predict_single(model, vectorizer, table_data, args.device)
        
        print(f"\n{'='*70}")
        print(f"PREDICTED QUALITY SCORE: {score:.4f} ({score*100:.2f}%)")
        print(f"{'='*70}")
        
        result = {
            'filename': args.input.name,
            'predicted_score': score
        }
        
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to: {args.output}")
    
    # Batch prediction
    elif args.input.is_dir():
        print(f"\nBatch prediction for directory: {args.input}")
        
        json_files = sorted(args.input.glob('*.json'))
        json_files = [f for f in json_files if f.name != 'processing_stats.json']
        
        print(f"Found {len(json_files)} JSON files")
        
        results = []
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                table_data = json.load(f)
            
            score = predict_single(model, vectorizer, table_data, args.device)
            
            results.append({
                'filename': json_file.name,
                'predicted_score': score
            })
            
            print(f"  {json_file.name}: {score:.4f}")
        
        # Calculate statistics
        scores = [r['predicted_score'] for r in results]
        stats = {
            'num_files': len(results),
            'average_score': np.mean(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'std_score': np.std(scores)
        }
        
        print(f"\n{'='*70}")
        print(f"BATCH PREDICTION SUMMARY")
        print(f"{'='*70}")
        print(f"Files:        {stats['num_files']}")
        print(f"Average:      {stats['average_score']:.4f}")
        print(f"Min:          {stats['min_score']:.4f}")
        print(f"Max:          {stats['max_score']:.4f}")
        print(f"Std Dev:      {stats['std_score']:.4f}")
        print(f"{'='*70}")
        
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output_data = {
                'statistics': stats,
                'predictions': results
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    else:
        print(f"Error: {args.input} is neither a file nor a directory")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

