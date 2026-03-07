"""Simple MLP for tabular classification."""
from __future__ import annotations

import torch
import torch.nn as nn

from mini_networks.core.blocks.mlp import MLP


class TabularMLP(nn.Module):
    def __init__(self, n_features: int = 4, n_classes: int = 3, hidden: int = 64):
        super().__init__()
        self.net = MLP(n_features, hidden, n_classes, layers=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TabularLinear(nn.Module):
    def __init__(self, n_features: int = 4, n_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TabularTransformer(nn.Module):
    """Tokenize features and run a small Transformer encoder."""

    def __init__(self, n_features: int = 4, n_classes: int = 3, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F] -> [B, F, 1] -> [B, F, D]
        x = self.proj(x.unsqueeze(-1))
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.cls(pooled)
