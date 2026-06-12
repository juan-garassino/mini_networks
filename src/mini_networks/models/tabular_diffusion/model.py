"""MLP denoiser for diffusion over tabular rows — DDPM without the UNet.

Key idea: diffusion is not an image technique; it works on any continuous vector.
A row of n_features numbers is noised by the same forward process as a picture,
    x_t = sqrt(a_bar_t) * x_0 + sqrt(1 - a_bar_t) * eps,
and a denoiser eps_theta(x_t, t) is trained with the same MSE-to-noise objective,
||eps - eps_theta(x_t, t)||^2. Since rows have no spatial structure, the
convolutional UNet is replaced by a plain MLP.

This implementation: TabularDenoiser maps a noisy row through Linear(n_features=4
→ hidden_dim=128) → ReLU → Linear(128 → 128) → ReLU → Linear(128 → 4), predicting
the noise vector at the input's own shape. Sampling reuses the shared
NoiseScheduler to walk the reverse chain over rows instead of pixels.

Deliberately simplified — and worth noticing: the timestep t is accepted but
ignored (no time embedding), so the network cannot tell how noisy its input is
and must learn an average denoiser across all noise levels; that costs sample
quality and is the first thing to fix (add a sinusoidal or learned t embedding).
Compared with TabDDPM (Kotelnikov et al. 2022) there is also no handling of
categorical columns (multinomial diffusion), no per-column normalisation logic,
and no quantile transforms — inputs are assumed to be already-scaled floats.
"""
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
