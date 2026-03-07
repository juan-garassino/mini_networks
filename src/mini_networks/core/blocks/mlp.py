"""Reusable MLP blocks."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int = 2):
        super().__init__()
        net = []
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            net.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                net.append(nn.ReLU())
        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
