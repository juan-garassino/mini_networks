"""Vanilla GAN for MNIST: MLP Generator + MLP Discriminator.

Architecture (from legacy/002-adversarial):
  Generator:     noise [B, 100] → 256 → 512 → 1024 → 784 (Tanh) → image [B, 1, 28, 28]
  Discriminator: image [B, 1, 28, 28] → flatten → 1024 → 512 → 256 → 1 (Sigmoid)

Training uses standard GAN losses (BCELoss) with separate Adam optimizers.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    """MLP generator: maps latent noise → flattened image, reshaped to [B, 1, 28, 28]."""

    def __init__(self, latent_dim: int = 100, image_size: int = 28, in_channels: int = 1):
        super().__init__()
        out_dim = in_channels * image_size * image_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),  nn.LeakyReLU(0.2),
            nn.Linear(256, 512),         nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),        nn.LeakyReLU(0.2),
            nn.Linear(1024, out_dim),    nn.Tanh(),
        )
        self.image_size = image_size
        self.in_channels = in_channels

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, latent_dim] → [B, C, H, W]."""
        x = self.net(z)
        return x.view(z.size(0), self.in_channels, self.image_size, self.image_size)


class Discriminator(nn.Module):
    """MLP discriminator: image [B, C, H, W] → real/fake probability [B, 1]."""

    def __init__(self, image_size: int = 28, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        in_dim = in_channels * image_size * image_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(1024, 512),    nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(512, 256),     nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(256, 1),       nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] → [B, 1] probability of being real."""
        return self.net(x.view(x.size(0), -1))


def gan_d_loss(
    discriminator: Discriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    criterion: nn.BCELoss,
) -> torch.Tensor:
    """Discriminator loss: real→1, fake→0."""
    B = real.size(0)
    ones  = torch.ones(B, 1, device=real.device)
    zeros = torch.zeros(B, 1, device=real.device)
    loss_real = criterion(discriminator(real), ones)
    loss_fake = criterion(discriminator(fake.detach()), zeros)
    return loss_real + loss_fake


def gan_g_loss(
    discriminator: Discriminator,
    fake: torch.Tensor,
    criterion: nn.BCELoss,
) -> torch.Tensor:
    """Generator loss: fool discriminator — fake should look real (target=1)."""
    B = fake.size(0)
    ones = torch.ones(B, 1, device=fake.device)
    return criterion(discriminator(fake), ones)
