"""Tabular preprocessing utilities."""
from __future__ import annotations

import torch


def normalize_batch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize features per-batch (zero mean, unit variance)."""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + eps
    return (x - mean) / std
