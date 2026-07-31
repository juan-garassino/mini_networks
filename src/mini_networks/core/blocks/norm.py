"""Shared normalization blocks."""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm: x * w / sqrt(mean(x^2) + eps) — no mean
    subtraction, no bias (torch<2.4 has no nn.RMSNorm).

    Consumers: models/kimi (KDA head norm, AttnRes keys, LatentMoE),
    models/deepseek (mHC, compressed-attention q/kv norms, MTP).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
