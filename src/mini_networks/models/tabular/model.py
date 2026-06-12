"""Three tabular classifiers of increasing capacity: linear, MLP, and Transformer.

Key idea: tabular data is the one domain where deep learning must earn its keep
against trivial baselines. This file stacks the ladder explicitly so the gap (or
lack of one) is measurable on the same features.

This implementation (defaults n_features=4, n_classes=3, e.g. Iris-sized data):
TabularLinear is multinomial logistic regression, a single Linear(4 → 3) — the
floor any deeper model has to beat. TabularMLP wraps the shared core MLP block
as 4 → 64 → 64 → 3 (layers=3). TabularTransformer tokenises each scalar feature
independently — x [B, F] → unsqueeze to [B, F, 1] → Linear(1 → d_model=64) giving
one 64-dim token per column — then runs a 2-layer nn.TransformerEncoder (4 heads)
so features can attend to each other, mean-pools over the F tokens, and classifies
with Linear(64 → 3).

Key equation: the linear baseline is p(y | x) = softmax(W x + b); the Transformer
variant adds inter-feature attention softmax(Q K^T / sqrt(d)) V on top of the
same inputs.

Deliberately simplified vs FT-Transformer (Gorishniy et al. 2021): one shared
Linear(1, d) tokenizer instead of per-feature weights and biases, no [CLS] token
(mean pooling instead), no categorical-feature embeddings, and no positional or
column-identity encoding — every column passes through the identical projection.
"""
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
