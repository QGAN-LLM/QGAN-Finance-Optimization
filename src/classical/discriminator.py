"""
Classical discriminator network for QGAN.
"""
import torch
import torch.nn as nn
from typing import List

class ClassicalDiscriminator(nn.Module):
    """Classical discriminator for distinguishing real vs generated data."""
    
    def __init__(self, 
                 input_size: int, 
                 hidden_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.2):
        """
        Initialize discriminator.
        
        Args:
            input_size: Size of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        layers = []
        prev_dim = input_size
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator."""
        return self.model(x)