"""Lightweight encoders for multimodal experiments."""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class VisionCNNEncoder(nn.Module):
    """Small CNN encoder that returns a pooled embedding."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.proj = nn.Linear(64 * 7 * 7, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.proj(h)


class VisionPatchEncoder(nn.Module):
    """Patchify grayscale images into token embeddings."""

    def __init__(self, patch_size: int = 4, d_model: int = 128, image_size: int = 28):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x + self.pos


class TextEncoder(nn.Module):
    """Simple token + positional embedding with a Transformer encoder."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 64,
    ):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token(tokens)
        x = x + self.pos[:, : x.size(1), :]
        return self.encoder(x)


def pool_sequence(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    if mode == "mean":
        return x.mean(dim=1)
    if mode == "cls":
        return x[:, 0]
    raise ValueError(f"Unknown pool mode: {mode}")


class AudioConvEncoder(nn.Module):
    """1D conv encoder for waveforms. Returns token sequence."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, d_model, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 5, stride=2, padding=2),
            nn.ReLU(),
        )

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        # wave: [B, 1, T] -> tokens [B, T', D]
        x = self.conv(wave)
        return x.transpose(1, 2)


class TabularFeatureEncoder(nn.Module):
    """Project tabular features into token embeddings."""

    def __init__(self, n_features: int = 8, d_model: int = 128):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F] -> tokens [B, F, D]
        x = x.unsqueeze(-1)
        return self.proj(x)
