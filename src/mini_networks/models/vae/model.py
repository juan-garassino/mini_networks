"""Simple convolutional VAE for 28x28 grayscale images.

A Variational Autoencoder (Kingma & Welling, 2013) is an autoencoder whose
bottleneck is a probability distribution, not a point: the encoder predicts
q(z|x) = N(mu, sigma^2) and the decoder reconstructs x from a sample z.
Training maximises the ELBO:

    ELBO = E_q[log p(x|z)] - KL(q(z|x) || N(0, I))

i.e. reconstruct well while keeping the posterior close to a standard
normal prior — which is what makes z ~ N(0, I) decodable into new samples.
Sampling stays differentiable via the reparameterisation trick:
z = mu + sigma * eps, with eps ~ N(0, I).

This implementation (latent_dim=32):

    encode:  [B,1,28,28] -> Conv s=2 (32) -> Conv s=2 (64) -> [B,64,7,7]
             -> flatten(3136) -> fc_mu / fc_logvar -> [B,32] each
    decode:  z [B,32] -> Linear(32->3136) -> reshape [B,64,7,7]
             -> ConvT s=2 (32) -> ConvT s=2 (1) -> Sigmoid -> [B,1,28,28]

vae_loss combines mean MSE reconstruction with the closed-form Gaussian KL
KL = -0.5 * mean(1 + logvar - mu^2 - exp(logvar)); beta scales the KL term
(beta-VAE style: higher beta -> more regular latent, blurrier outputs).

Deliberately simplified: MSE instead of the Bernoulli log-likelihood the
paper uses for binarised MNIST, a tiny 2-conv encoder/decoder, mean
reduction over both terms (so the recon/KL balance differs from the strict
per-image ELBO), and no KL warm-up or free-bits tricks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.enc_out_dim = 64 * 7 * 7
        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, self.enc_out_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(z.size(0), 64, 7, 7)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl
