"""REINFORCE policy network for MazeEnv — the simplest policy-gradient method.

Key idea: skip value functions entirely and adjust the policy directly. The
policy-gradient theorem gives
    grad J(theta) = E[ grad log pi_theta(a_t | s_t) * R_t ],
so actions followed by high return get their log-probability pushed up, and
actions followed by low return get pushed down. No TD bootstrapping, no critic —
just Monte Carlo returns from complete episodes.

This implementation: PolicyNet is a two-layer MLP, state_size → hidden_dim=64
(ReLU) → n_actions=4, returning log-probabilities via log_softmax. The trainer
samples actions from Categorical(logits), plays a full episode, computes
discounted returns G_t = r_t + gamma * G_{t+1}, normalises them to zero mean and
unit variance within the episode (a cheap variance-reduction baseline), and
minimises loss = -sum_t log pi(a_t | s_t) * G_t with Adam.

Deliberately simplified vs actor-critic methods (and Williams 1992 extensions):
no learned baseline or critic — per-episode return normalisation stands in for
one; pure Monte Carlo returns mean high gradient variance and slow learning; one
episode per update with no entropy bonus, so exploration relies entirely on the
stochasticity of the softmax policy.
"""
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
