#!/usr/bin/env python3
"""
Train MLP regressor to predict table extraction quality scores.

This script loads the table-extraction-evaluation dataset from Hugging Face,
extracts Sentence Transformer features from the generated table JSONs, and trains a simple
MLP to predict similarity scores without requiring ground truth.
"""

import argparse
import json
import sys
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from mlp_regressor import MLPRegressor, ImprovedMLPRegressor, DeepMLPRegressor, TableQualityDataset
from utils.table_features import extract_all_features


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
        # Convert generated table to JSON string for Sentence Transformer encoding
        texts.append(json.dumps(sample['generated']))
        labels.append(sample['similarity_score'])
        ids.append(sample['id'])
    
    return texts, labels, ids


def prepare_features(
    texts_train,
    texts_val,
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    save_path: Path = None,
    use_hybrid_features: bool = False
):
    """
    Extract features from table JSON texts.
    
    Args:
        texts_train: Training texts (JSON strings)
        texts_val: Validation texts (JSON strings)
        model_name: Name of the sentence-transformers model
        save_path: Path to save the model configuration
        use_hybrid_features: If True, use structure + text + embeddings; else just embeddings
    
    Returns:
        X_train, X_val: Feature matrices
        model_name: Name of the sentence transformer model
    """
    if use_hybrid_features:
        print("Using hybrid features (structure + text stats + embeddings)...")
        print(f"Loading Sentence Transformer model: {model_name}...")
        embedder = SentenceTransformer(model_name)
        
        print(f"Extracting hybrid features from {len(texts_train)} training samples...")
        X_train = []
        for text in tqdm(texts_train, desc="Training features"):
            features = extract_all_features(
                text,
                sentence_transformer=embedder,
                normalize_embeddings=False
            )
            X_train.append(features)
        X_train = np.array(X_train)
        
        print(f"Extracting hybrid features from {len(texts_val)} validation samples...")
        X_val = []
        for text in tqdm(texts_val, desc="Validation features"):
            features = extract_all_features(
                text,
                sentence_transformer=embedder,
                normalize_embeddings=False
            )
            X_val.append(features)
        X_val = np.array(X_val)
        
        print(f"Hybrid feature matrix shape (before standardization): {X_train.shape}")
        print(f"  - Structure features: ~30")
        print(f"  - Text features: ~10")
        print(f"  - Embedding dimension: {X_train.shape[1] - 40}")
    else:
        print(f"Loading Sentence Transformer model: {model_name}...")
        embedder = SentenceTransformer(model_name)
        
        print(f"Encoding training texts...")
        X_train = embedder.encode(
            texts_train,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=False  # We'll standardize instead
        )
        
        print(f"Encoding validation texts...")
        X_val = embedder.encode(
            texts_val,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        
        print(f"Feature matrix shape (before standardization): {X_train.shape}")
        print(f"Embedding dimension: {X_train.shape[1]}")
    
    # Standardize features (zero mean, unit variance)
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    print(f"After standardization - Mean: {X_train.mean():.4f}, Std: {X_train.std():.4f}")
    
    # Save model configuration
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model name (model can be reloaded from name)
        model_config_path = save_path.parent / 'sentence_transformer_config.json'
        with open(model_config_path, 'w') as f:
            json.dump({
                'model_name': model_name,
                'use_hybrid_features': use_hybrid_features
            }, f, indent=2)
        print(f"Sentence Transformer config saved to: {model_config_path}")
        
        # Save scaler
        scaler_path = save_path.parent / 'feature_scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Feature scaler saved to: {scaler_path}")
    
    return X_train, X_val, model_name


def train_epoch(model, dataloader, optimizer, loss_fn, device, max_grad_norm=1.0):
    """Train for one epoch with gradient clipping."""
    model.train()
    total_loss = 0
    
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        
        pred = model(xb)
        loss = loss_fn(pred, yb)
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, loss_fn, device):
    """
    Evaluate model on validation set with comprehensive metrics.
    
    This function computes multiple metrics suitable for evaluating percentage/score predictions:
    - MSE, MAE, RMSE: Standard regression metrics
    - MAPE: Mean Absolute Percentage Error - normalizes errors by ground truth magnitude
    - R²: Coefficient of Determination - measures how well predictions explain variance
    - Accuracy (±1%, ±5%, ±10%): Percentage of predictions within tolerance thresholds
    - Median AE: More robust to outliers than MAE
    
    These metrics are more appropriate than exact matching for percentage comparisons
    because they account for the relative magnitude of errors and provide tolerance-based
    assessments that are more meaningful for continuous scores in [0, 1] range.
    """
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
    
    # Standard regression metrics
    mse = total_loss / len(dataloader)
    mae = torch.mean(torch.abs(all_preds - all_targets)).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    
    # Percentage-based metrics (better for scores in [0, 1] range)
    # MAPE (Mean Absolute Percentage Error) - normalized by ground truth
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    mape = torch.mean(torch.abs((all_targets - all_preds) / (all_targets + epsilon))) * 100
    mape = mape.item()
    
    # R² Score (Coefficient of Determination) - measures explained variance
    ss_res = torch.sum((all_targets - all_preds) ** 2)
    ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + epsilon))
    r2 = r2.item()
    
    # Tolerance-based accuracy (percentage within ±1%, ±5%, ±10%)
    abs_diff = torch.abs(all_preds - all_targets)
    acc_1pct = (abs_diff <= 0.01).float().mean().item() * 100
    acc_5pct = (abs_diff <= 0.05).float().mean().item() * 100
    acc_10pct = (abs_diff <= 0.10).float().mean().item() * 100
    
    # Median Absolute Error (more robust to outliers)
    median_ae = torch.median(abs_diff).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,  # Mean Absolute Percentage Error
        'r2': r2,  # R² Score
        'acc_1pct': acc_1pct,  # % predictions within ±0.01 of truth
        'acc_5pct': acc_5pct,  # % predictions within ±0.05 of truth
        'acc_10pct': acc_10pct,  # % predictions within ±0.10 of truth
        'median_ae': median_ae  # Median absolute error
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
    weight_decay: float = 1e-4,
    device: str = 'cpu',
    checkpoint_dir: Path = None,
    use_improved_model: bool = False,
    use_deep_model: bool = False,
    hidden_dims: list = None,
    use_layer_norm: bool = False
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
        use_improved_model: Whether to use ImprovedMLPRegressor
        use_deep_model: Whether to use DeepMLPRegressor (overrides use_improved_model)
        hidden_dims: List of hidden dimensions for DeepMLPRegressor
        use_layer_norm: Whether to use LayerNorm in DeepMLPRegressor
    
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
    if use_deep_model:
        model = DeepMLPRegressor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_residual=True,
            use_layer_norm=use_layer_norm
        )
        print(f"Using DeepMLPRegressor with {len(model.layers)} hidden layers")
        if hidden_dims:
            print(f"  Hidden dimensions: {hidden_dims}")
        else:
            print(f"  Using default dimensions: {[m.out_features for m in model.layers]}")
    elif use_improved_model:
        model = ImprovedMLPRegressor(
            input_dim=input_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            dropout_rate=dropout_rate,
            use_residual=True
        )
        print("Using ImprovedMLPRegressor with residual connections")
    else:
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
    
    # Loss and optimizer with L2 regularization to combat overfitting
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler - reduce LR when validation plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    history = {
        'train_loss': [],
        'val_metrics': []
    }
    
    best_val_mae = float('inf')
    patience_counter = 0
    early_stop_patience = 15
    prev_lr = learning_rate
    
    print(f"\nTraining on {device}")
    print(f"Model architecture: {input_dim} -> {hidden_dim1} -> {hidden_dim2} -> 1")
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print(f"Dropout: {dropout_rate}, Weight decay (L2): {weight_decay}")
    print(f"Early stopping patience: {early_stop_patience}\n")
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_dl, optimizer, loss_fn, device)
        
        # Validate
        val_metrics = evaluate(model, val_dl, loss_fn, device)
        
        history['train_loss'].append(train_loss)
        history['val_metrics'].append(val_metrics)
        
        # Step scheduler and check if LR changed
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['mae'])
        new_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss:  {train_loss:.4f}")
        print(f"  Val MAE:     {val_metrics['mae']:.4f}")
        print(f"  Val RMSE:    {val_metrics['rmse']:.4f}")
        print(f"  Val R²:      {val_metrics['r2']:.4f}")
        print(f"  Val MAPE:    {val_metrics['mape']:.2f}%")
        print(f"  Acc (±1%):   {val_metrics['acc_1pct']:.1f}%")
        print(f"  Acc (±5%):   {val_metrics['acc_5pct']:.1f}%")
        print(f"  Acc (±10%):  {val_metrics['acc_10pct']:.1f}%")
        
        # Log LR changes
        if new_lr != old_lr:
            print(f"  📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        # Save best model
        if checkpoint_dir and val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            patience_counter = 0
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
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n⚠ Early stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
                break
    
    return model, history


def plot_training_history(history, save_dir: Path):
    """
    Plot training history: loss and metrics vs epochs.
    
    Args:
        history: Dictionary with 'train_loss' and 'val_metrics' lists
        save_dir: Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Extract validation metrics
    val_mae = [m['mae'] for m in history['val_metrics']]
    val_rmse = [m['rmse'] for m in history['val_metrics']]
    val_mse = [m['mse'] for m in history['val_metrics']]
    val_r2 = [m['r2'] for m in history['val_metrics']]
    val_acc_1pct = [m['acc_1pct'] for m in history['val_metrics']]
    val_acc_5pct = [m['acc_5pct'] for m in history['val_metrics']]
    val_acc_10pct = [m['acc_10pct'] for m in history['val_metrics']]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Training Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss (MSE)')
    axes[0].plot(epochs, val_mse, 'r--', linewidth=2, label='Val Loss (MSE)')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')  # Log scale for better visualization
    
    # Plot 2: Validation Error Metrics
    axes[1].plot(epochs, val_mae, 'g-', linewidth=2, label='MAE', marker='o', markersize=4)
    axes[1].plot(epochs, val_rmse, 'orange', linewidth=2, label='RMSE', marker='s', markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Error', fontsize=12)
    axes[1].set_title('Validation Error Metrics', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: R² and Accuracy Metrics
    ax3_1 = axes[2]
    ax3_2 = ax3_1.twinx()
    
    # R² on left axis
    line1 = ax3_1.plot(epochs, val_r2, 'purple', linewidth=2, label='R² Score', marker='d', markersize=4)
    ax3_1.set_xlabel('Epoch', fontsize=12)
    ax3_1.set_ylabel('R² Score', fontsize=12, color='purple')
    ax3_1.tick_params(axis='y', labelcolor='purple')
    ax3_1.grid(True, alpha=0.3)
    
    # Accuracy on right axis
    line2 = ax3_2.plot(epochs, val_acc_1pct, 'green', linewidth=2, label='Acc (±1%)', marker='^', markersize=4)
    line3 = ax3_2.plot(epochs, val_acc_5pct, 'blue', linewidth=2, label='Acc (±5%)', marker='o', markersize=4)
    line4 = ax3_2.plot(epochs, val_acc_10pct, 'cyan', linewidth=2, label='Acc (±10%)', marker='s', markersize=4)
    ax3_2.set_ylabel('Accuracy (%)', fontsize=12, color='blue')
    ax3_2.tick_params(axis='y', labelcolor='blue')
    ax3_2.set_ylim([0, 100])
    
    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax3_1.legend(lines, labels, fontsize=10, loc='best')
    ax3_1.set_title('R² and Accuracy Metrics', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = save_dir / 'training_history.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Training plot saved to: {plot_path}")
    plt.close()
    
    # Also create a separate detailed loss plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=3)
    ax.plot(epochs, val_mse, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Add best epoch marker
    best_epoch = np.argmin(val_mae) + 1
    best_val_loss = val_mse[best_epoch - 1]
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(best_epoch, best_val_loss, f'  Best (epoch {best_epoch})', 
            fontsize=10, color='green', verticalalignment='bottom')
    
    plt.tight_layout()
    loss_plot_path = save_dir / 'loss_vs_epoch.png'
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Loss plot saved to: {loss_plot_path}")
    plt.close()


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
        '--model-name',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Sentence Transformer model name (default: sentence-transformers/all-MiniLM-L6-v2)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size (default: 64)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='Learning rate (default: 0.0003)'
    )
    parser.add_argument(
        '--hidden-dim1',
        type=int,
        default=512,
        help='First hidden layer size (default: 512)'
    )
    parser.add_argument(
        '--hidden-dim2',
        type=int,
        default=256,
        help='Second hidden layer size (default: 256)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.0,
        help='Dropout rate (default: 0.0)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='L2 regularization strength (default: 1e-4)'
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
    parser.add_argument(
        '--use-hybrid-features',
        action='store_true',
        help='Use hybrid features (structure + text stats + embeddings) instead of just embeddings'
    )
    parser.add_argument(
        '--use-improved-model',
        action='store_true',
        help='Use ImprovedMLPRegressor with residual connections'
    )
    parser.add_argument(
        '--use-deep-model',
        action='store_true',
        help='Use DeepMLPRegressor with multiple layers (overrides --use-improved-model)'
    )
    parser.add_argument(
        '--hidden-dims',
        type=int,
        nargs='+',
        default=None,
        help='Hidden layer dimensions for DeepMLPRegressor (e.g., --hidden-dims 1024 512 256 128)'
    )
    parser.add_argument(
        '--use-layer-norm',
        action='store_true',
        help='Use LayerNorm in addition to BatchNorm (for DeepMLPRegressor)'
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
    X_train, X_val, model_name = prepare_features(
        texts_train,
        texts_val,
        model_name=args.model_name,
        save_path=args.output_dir / 'embeddings',
        use_hybrid_features=args.use_hybrid_features
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
        weight_decay=args.weight_decay,
        device=args.device,
        checkpoint_dir=args.output_dir,
        use_improved_model=args.use_improved_model,
        use_deep_model=args.use_deep_model,
        hidden_dims=args.hidden_dims,
        use_layer_norm=args.use_layer_norm
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
    
    # Plot training history
    plot_training_history(history, args.output_dir)
    
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

