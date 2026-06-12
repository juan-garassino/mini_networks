"""Three ways to fuse two modalities, from crudest to most expressive.

Key idea: once each modality is encoded, the design question is how their
representations interact. This file implements the standard ladder: concatenate
pooled vectors, gate between pooled vectors, or let one token sequence attend to
the other before pooling. Each strategy exposes the same call shape so they are
swappable in the encoders one level up.

This implementation (d_model=128 default): ConcatFusion concatenates two pooled
vectors [B, 128] each into [B, 256] and projects back with one Linear — modality
interaction happens only through that single matrix. GatedFusion computes
g = sigmoid(W [a; b]) and returns g * a + (1 - g) * b, a learned per-dimension
convex blend that can dynamically favour one modality. CrossAttentionBlock is one
round of attention where the query sequence attends to the context sequence:
softmax(Q_query K_ctx^T / sqrt(d)) V_ctx (4 heads), residual + LayerNorm, then a
4x-wide GELU FFN with its own residual. CrossAttentionFusion wraps that block and
pools the result to [B, 128] by mean or first-token ("cls") readout.

Deliberately simplified vs production fusion stacks: a single cross-attention
layer, one direction only (no co-attention where both sides query each other),
no gating between the attention output and the query stream (cf. Flamingo's tanh
gates), and the FFN residual is added without a final LayerNorm — fine at this
depth, sloppy if you stack it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """Cross-attention: query attends to context."""

    def __init__(self, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(query, context, context, need_weights=False)
        x = self.ln(query + attn_out)
        return self.ffn(x) + x


class ConcatFusion(nn.Module):
    """Concatenate two pooled vectors and project."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim * 2, out_dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([a, b], dim=-1))


class GatedFusion(nn.Module):
    """Learned gate between two pooled vectors."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([a, b], dim=-1)))
        return g * a + (1 - g) * b


class CrossAttentionFusion(nn.Module):
    """Cross-attend two sequences and pool."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, pool: str = "mean"):
        super().__init__()
        self.cross = CrossAttentionBlock(d_model=d_model, n_heads=n_heads)
        self.pool = pool

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.cross(query, context)
        if self.pool == "mean":
            return x.mean(dim=1)
        if self.pool == "cls":
            return x[:, 0]
        raise ValueError(f"Unknown pool mode: {self.pool}")
