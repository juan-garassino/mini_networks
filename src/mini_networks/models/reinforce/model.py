"""REINFORCE policy network for MazeEnv."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, state_size: int, hidden_dim: int = 64, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.log_softmax(logits, dim=-1)
