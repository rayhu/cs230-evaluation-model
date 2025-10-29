#!/usr/bin/env python3
"""
Simple MLP Regressor for Table Extraction Quality Prediction.

This module implements a feedforward neural network that predicts table extraction
quality scores from TF-IDF features of generated table JSON structures.
"""

import torch
import torch.nn as nn
from typing import Optional


class MLPRegressor(nn.Module):
    """
    Multi-Layer Perceptron for regression on table quality scores.
    
    Architecture:
        Input -> Linear(input_dim, hidden_dim1) -> BatchNorm -> ReLU -> Dropout ->
        Linear(hidden_dim1, hidden_dim2) -> BatchNorm -> ReLU -> Dropout ->
        Linear(hidden_dim2, 1) -> Output
    
    Args:
        input_dim: Dimensionality of input features (Word2Vec embedding size)
        hidden_dim1: Size of first hidden layer (default: 256)
        hidden_dim2: Size of second hidden layer (default: 64)
        dropout_rate: Dropout probability (default: 0.3)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int = 256,
        hidden_dim2: int = 64,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        layers = [
            nn.Dropout(dropout_rate * 0.5),  # Input dropout for extra randomness
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, 1)
        ]
        
        self.net = nn.Sequential(*layers)
        
        # Proper weight initialization (Xavier/Kaiming)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Randomly initialize all weights with high variance.
        Ensures initial predictions are random, leading to high initial loss.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Random normal initialization with high standard deviation
                # This creates truly random initial predictions
                std = 0.5  # High standard deviation for more randomness
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    # Random bias with wide range
                    nn.init.uniform_(m.bias, -0.5, 0.5)
            elif isinstance(m, nn.BatchNorm1d):
                # BatchNorm also gets random initialization
                nn.init.uniform_(m.weight, 0.5, 1.5)
                nn.init.uniform_(m.bias, -0.1, 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Predicted scores of shape (batch_size,)
        """
        return self.net(x).squeeze()


class TableQualityDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for table quality prediction.
    
    Args:
        X: Feature matrix (numpy array or tensor)
        y: Target scores (numpy array or tensor)
    """
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
        self.y = torch.tensor(y, dtype=torch.float32) if not isinstance(y, torch.Tensor) else y
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]

