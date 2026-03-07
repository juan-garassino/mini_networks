"""Transformer token classifier."""
from __future__ import annotations

import torch
import torch.nn as nn


class TokenClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, seq_len: int):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 2)  # vowel vs other

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token(x) + self.pos(pos)
        h = self.encoder(h)
        return self.head(h)
