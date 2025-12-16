#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models.py
Defines DeepMLP and DeepMLPEnsemble architectures for feature-based
histopathology classification and ensemble learning.

Usage:
    from models import DeepMLP, DeepMLPEnsemble
"""

import torch
import torch.nn as nn


class DeepMLP(nn.Module):
    """Single-stream Multi-Layer Perceptron for patch-level classification."""
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class DeepMLPEnsemble(nn.Module):
    """
    Concatenation-based ensemble MLP combining features from
    multiple foundation models (e.g., Musk, Hoptimus, Conch).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


if __name__ == "__main__":
    # Quick sanity check
    model = DeepMLP(input_dim=1024)
    ens = DeepMLPEnsemble(input_dim=3072)
    dummy = torch.randn(8, 1024)
    dummy_ens = torch.randn(8, 3072)
    print("DeepMLP output:", model(dummy).shape)
    print("Ensemble output:", ens(dummy_ens).shape)
