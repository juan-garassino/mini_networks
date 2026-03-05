"""Variational Autoencoder for latent diffusion on MNIST.

Compresses 1×28×28 → 4×7×7 latent space, enabling diffusion to run
at 1/16 the pixel cost. Architecture mirrors legacy reference.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Encoder: 1×28×28 → mu/logvar each [latent_channels×7×7]
    Decoder: latent [latent_channels×7×7] → 1×28×28

    The latent is obtained via reparameterisation: z = mu + eps * exp(0.5 * logvar)
    """

    def __init__(self, latent_channels: int = 4):
        super().__init__()
        self.latent_channels = latent_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, latent_channels * 2, 1),  # → [2*lc, 7, 7]
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 3, padding=1), nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, logvar) each of shape [B, latent_channels, H/4, W/4]."""
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent [B, lc, H, W] → image [B, 1, 28, 28] in [-1, 1]."""
        out = self.decoder(z)
        # Ensure output matches 28×28 exactly (ConvTranspose may over/undershoot by 1px)
        return F.interpolate(out, size=(28, 28), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar

    @property
    def latent_size(self) -> tuple[int, int, int]:
        """Shape of one latent sample (C, H, W)."""
        return (self.latent_channels, 7, 7)


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1e-3,
) -> torch.Tensor:
    """ELBO loss: reconstruction (MSE) + KL divergence."""
    recon_loss = F.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl
