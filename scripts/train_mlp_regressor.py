#!/usr/bin/env python3
"""
Train MLP regressor to predict table extraction quality scores.

This script loads the table-extraction-evaluation dataset from Hugging Face,
extracts Word2Vec features from the generated table JSONs, and trains a simple
MLP to predict similarity scores without requiring ground truth.
"""

import argparse
import json
import sys
from pathlib import Path
import pickle
import re
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
from tqdm import tqdm
from gensim.models import Word2Vec

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from mlp_regressor import MLPRegressor, TableQualityDataset


def load_data(split: str = 'train', limit: int = None):
    """
    Load dataset from Hugging Face.
    
    Args:
        split: Dataset split to load ('train' or 'test')
        limit: Maximum number of samples to load (for testing)
    
    Returns:
        texts: List of JSON strings representing generated tables
        labels: List of similarity scores
        ids: List of sample IDs
    """
    print(f"Loading dataset split: {split}")
    dataset = load_dataset("rayhu/table-extraction-evaluation", split=split)
    print(f"Dataset length: {len(dataset)}")
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    texts = []
    labels = []
    ids = []
    
    print(f"Processing {len(dataset)} samples...")
    for sample in tqdm(dataset):
        # Convert generated table to JSON string for Word2Vec
        texts.append(json.dumps(sample['generated']))
        labels.append(sample['similarity_score'])
        ids.append(sample['id'])
    
    return texts, labels, ids


def tokenize_json(text: str):
    """Tokenize JSON text into words."""
    # Extract alphanumeric tokens, including numbers
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def prepare_features(
    texts_train,
    texts_val,
    vector_size: int = 300,
    window: int = 5,
    min_count: int = 2,
    workers: int = 4,
    save_path: Path = None
):
    """
    Extract Word2Vec features from texts.
    
    Args:
        texts_train: Training texts
        texts_val: Validation texts
        vector_size: Dimensionality of word vectors
        window: Context window size for Word2Vec
        min_count: Minimum word frequency
        workers: Number of worker threads
        save_path: Path to save the Word2Vec model
    
    Returns:
        X_train, X_val: Feature matrices
        w2v_model: Trained Word2Vec model
    """
    print(f"Tokenizing texts...")
    # Tokenize all texts
    train_tokens = [tokenize_json(text) for text in tqdm(texts_train, desc="Tokenizing train")]
    val_tokens = [tokenize_json(text) for text in tqdm(texts_val, desc="Tokenizing val")]
    
    print(f"Training Word2Vec model (vector_size={vector_size}, window={window})...")
    # Train Word2Vec on training data
    w2v_model = Word2Vec(
        sentences=train_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=42,
        sg=1,  # Skip-gram model (better for small datasets)
        epochs=10
    )
    
    print(f"Converting documents to vectors...")
    # Convert documents to average word vectors
    def doc_to_vec(tokens, model):
        """Average word vectors for all words in document."""
        vectors = []
        for word in tokens:
            if word in model.wv:
                vectors.append(model.wv[word])
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            # Return zero vector if no words found
            return np.zeros(model.vector_size)
    
    X_train = np.array([doc_to_vec(tokens, w2v_model) for tokens in tqdm(train_tokens, desc="Train vectors")])
    X_val = np.array([doc_to_vec(tokens, w2v_model) for tokens in tqdm(val_tokens, desc="Val vectors")])
    
    print(f"Feature matrix shape (before normalization): {X_train.shape}")
    print(f"Vocabulary size: {len(w2v_model.wv)}")
    
    # Standardize features (zero mean, unit variance)
    # This prevents Word2Vec from being too predictive with random weights
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    print(f"After standardization - Mean: {X_train.mean():.4f}, Std: {X_train.std():.4f}")
    
    # Save models
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        w2v_model.save(str(save_path))
        print(f"Word2Vec model saved to: {save_path}")
        
        # Save scaler
        scaler_path = save_path.parent / 'feature_scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Feature scaler saved to: {scaler_path}")
    
    return X_train, X_val, w2v_model


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        
        pred = model(xb)
        loss = loss_fn(pred, yb)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, loss_fn, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            
            pred = model(xb)
            loss = loss_fn(pred, yb)
            
            total_loss += loss.item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
    
    all_preds = torch.tensor(all_preds)
    all_targets = torch.tensor(all_targets)
    
    mse = total_loss / len(dataloader)
    mae = torch.mean(torch.abs(all_preds - all_targets)).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    hidden_dim1: int = 256,
    hidden_dim2: int = 64,
    dropout_rate: float = 0.0,
    device: str = 'cpu',
    checkpoint_dir: Path = None
):
    """
    Train the MLP regressor.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for Adam optimizer
        hidden_dim1: First hidden layer size
        hidden_dim2: Second hidden layer size
        dropout_rate: Dropout rate (0 for no dropout)
        device: Device to train on ('cpu', 'cuda', or 'mps')
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        model: Trained model
        history: Training history
    """
    # Create datasets
    train_ds = TableQualityDataset(X_train, y_train)
    val_ds = TableQualityDataset(X_val, y_val)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    
    # Initialize model with random weights
    input_dim = X_train.shape[1]
    model = MLPRegressor(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        dropout_rate=dropout_rate
    )
    
    # Verify random initialization by checking initial predictions
    model.eval()
    with torch.no_grad():
        sample_input = torch.randn(1, input_dim)
        sample_output = model(sample_input)
        print(f"Sample random prediction (should be unpredictable): {sample_output.item():.4f}")
    
    model.to(device)
    model.train()  # Set back to training mode
    
    # Loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    history = {
        'train_loss': [],
        'val_metrics': []
    }
    
    best_val_mae = float('inf')
    
    print(f"\nTraining on {device}")
    print(f"Model architecture: {input_dim} -> {hidden_dim1} -> {hidden_dim2} -> 1")
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}\n")
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_dl, optimizer, loss_fn, device)
        
        # Validate
        val_metrics = evaluate(model, val_dl, loss_fn, device)
        
        history['train_loss'].append(train_loss)
        history['val_metrics'].append(val_metrics)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val MAE:    {val_metrics['mae']:.4f}")
        print(f"  Val RMSE:   {val_metrics['rmse']:.4f}")
        
        # Save best model
        if checkpoint_dir and val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': best_val_mae,
                'hyperparameters': {
                    'input_dim': input_dim,
                    'hidden_dim1': hidden_dim1,
                    'hidden_dim2': hidden_dim2,
                    'dropout_rate': dropout_rate
                }
            }, checkpoint_path)
            print(f"  → Saved best model (MAE: {best_val_mae:.4f})")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP regressor for table quality prediction"
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of training samples (for testing)'
    )
    parser.add_argument(
        '--vector-size',
        type=int,
        default=1024,
        help='Word2Vec embedding dimension (default: 300)'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=10,
        help='Word2Vec context window size (default: 5)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--hidden-dim1',
        type=int,
        default=1024,
        help='First hidden layer size (default: 256)'
    )
    parser.add_argument(
        '--hidden-dim2',
        type=int,
        default=512,
        help='Second hidden layer size (default: 64)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate (default: 0.3)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Validation split ratio (default: 0.2)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('experiments/mlp_regressor'),
        help='Output directory for models and results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to train on (default: cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42, use -1 for random seed each time)'
    )
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    import random
    import time
    
    # Use random seed if requested
    if args.seed == -1:
        args.seed = int(time.time()) % 100000
        print(f"Using random seed: {args.seed}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"Random seed set to: {args.seed}")
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load data
    texts, labels, ids = load_data(split='train', limit=args.limit)
    
    # Train/val split
    print(f"\nSplitting data (val_split={args.val_split})...")
    texts_train, texts_val, y_train, y_val = train_test_split(
        texts, labels, test_size=args.val_split, random_state=args.seed
    )
    
    # Extract features
    X_train, X_val, w2v_model = prepare_features(
        texts_train,
        texts_val,
        vector_size=args.vector_size,
        window=args.window,
        save_path=args.output_dir / 'word2vec_model.bin'
    )
    
    # Train model
    model, history = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        dropout_rate=args.dropout,
        device=args.device,
        checkpoint_dir=args.output_dir
    )
    
    # Save final model
    final_model_path = args.output_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': {
            'input_dim': X_train.shape[1],
            'hidden_dim1': args.hidden_dim1,
            'hidden_dim2': args.hidden_dim2,
            'dropout_rate': args.dropout
        }
    }, final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save training history
    history_path = args.output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Save configuration
    config_path = args.output_dir / 'config.json'
    config = vars(args)
    config['output_dir'] = str(config['output_dir'])  # Convert Path to string
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

