"""Minimal ViT for 28x28 grayscale images.

Vision Transformer (Dosovitskiy et al., 2020): treat an image as a sequence
of patch tokens and let self-attention relate every patch to every other in
a single layer — no convolutional locality prior at all. Attention is
softmax(Q K^T / sqrt(d)) V, so each patch aggregates information from all
patches, weighted by learned similarity.

This implementation:

    [B,1,28,28] -> Conv2d(k=4, s=4) patch embed -> 7x7 = 49 tokens of d=64
                -> prepend learnable [CLS] token  -> 50 tokens
                -> add learnable positional embedding (zeros-initialised)
                -> 4 x TransformerEncoderLayer (4 heads, FFN dim 128, dropout 0.1)
                -> LayerNorm on the CLS token -> Linear(64->10)

The strided conv is the standard trick for "flatten each PxP patch and
project it" in one op. Only the CLS token feeds the classifier head; the
positional embedding is what tells the model where each patch came from,
since attention itself is permutation-invariant.

Deliberately simplified vs the paper: tiny everything (ViT-Base uses
d=768, 12 layers, 12 heads, 16x16 patches on 224x224 images), no
large-scale pretraining — ViTs underperform CNNs when trained from scratch
on small data, which MNIST forgives — and stock PyTorch encoder layers
(post-norm by default) instead of the paper's pre-norm blocks.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MiniViT(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        mlp_dim: int = 128,
        num_classes: int = 10,
        image_size: int = 28,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.patch_size = patch_size
        n_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised CLS embedding, before the classifier head (DINO reuses this)."""
        x = self.patch_embed(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.norm(x[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))
