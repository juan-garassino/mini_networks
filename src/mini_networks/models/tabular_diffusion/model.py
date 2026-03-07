"""Simple MLP denoiser for tabular diffusion."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularDenoiser(nn.Module):
    def __init__(self, n_features: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t is ignored in this minimal model (can be added via embedding later)
        return self.net(x)
