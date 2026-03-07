"""Reusable attention blocks."""
from __future__ import annotations

import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)
