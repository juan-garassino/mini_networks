"""Small CNN classifier for 28x28 grayscale images."""
from __future__ import annotations

import torch
import torch.nn as nn

from mini_networks.core.blocks.cnn import ConvBNReLU


class SmallCNN(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(1, 32),
            nn.MaxPool2d(2),
            ConvBNReLU(32, 64),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
