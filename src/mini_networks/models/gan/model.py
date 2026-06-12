"""Vanilla GAN for MNIST: an MLP Generator and MLP Discriminator in a minimax game.

Key idea: two networks trained adversarially. The discriminator D learns to tell
real images from generated ones; the generator G learns to fool it. The original
objective is min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))] — at equilibrium
G's distribution matches the data and D outputs 1/2 everywhere.

This implementation: Generator maps noise z [B, 100] through Linear layers
100 → 256 → 512 → 1024 → 784 (LeakyReLU 0.2 between, Tanh out, so pixels live in
[-1, 1]) and reshapes to [B, 1, 28, 28]. Discriminator flattens the image and maps
784 → 1024 → 512 → 256 → 1 with LeakyReLU + Dropout 0.3 and a final Sigmoid
probability. Both losses are written in BCE form: gan_d_loss = BCE(D(real), 1) +
BCE(D(fake.detach()), 0) — note the detach so D's step does not touch G — and
gan_g_loss = BCE(D(fake), 1), i.e. the non-saturating heuristic -log D(G(z))
recommended in the original paper, which keeps gradients alive early when D wins.
The trainer alternates one Adam step for D, then one for G, each batch.

Deliberately simplified vs Goodfellow 2014 / DCGAN: fully-connected nets instead
of transposed/strided convolutions, no BatchNorm, Sigmoid + BCELoss rather than
the numerically safer BCEWithLogits, and none of the usual stabilisers (label
smoothing, spectral norm, feature matching) — mode collapse is left observable
on purpose.
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
