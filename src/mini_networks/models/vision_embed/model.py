"""Simple CNN encoder for embeddings."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_networks.core.blocks.cnn import ConvBNReLU


class VisionEmbedCNN(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(1, 32),
            nn.MaxPool2d(2),
            ConvBNReLU(32, 64),
            nn.MaxPool2d(2),
        )
        self.proj = nn.Linear(64 * 7 * 7, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        z = self.proj(h)
        return F.normalize(z, dim=-1)
