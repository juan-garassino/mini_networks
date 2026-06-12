"""GAN vs Diffusion side-by-side comparison composition.

Educational goal
----------------
  Train both a GAN and a DDPM on the same MNIST dataset under identical
  conditions, then compare their generated samples.

  Key differences this composition highlights:
    GAN:
      + Fast inference (single forward pass through G)
      + Sharp images when it works
      - Training instability; mode collapse risk
      - Adversarial min-max game is hard to balance

    DDPM:
      + Stable, well-behaved loss (MSE on noise)
      + Covers more of the data distribution
      - Slow inference (T denoising steps)

Pipeline
--------
  1. `train_gan(config, logger)`  — trains Generator + Discriminator
  2. `train_diffusion(config, logger)` — trains DDPM UNet
  3. `compare(config, n_samples)` — generates from both, returns side-by-side dict
  4. `sample_diversity(images)` — pixel-variance diversity metric

Both models use the same `config.data_root`, `config.fast_demo`, and
`config.effective_batch_size` so comparisons are fair.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.models.diffusion.model import UNet
from mini_networks.models.diffusion.scheduler import NoiseScheduler
from mini_networks.core.diffusion.sampling import sample_loop
from mini_networks.models.gan.model import Generator, Discriminator, gan_d_loss, gan_g_loss

import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class GANDiffusionConfig(BaseConfig):
    model_name: str = "gan_diffusion_comparison"

    # Shared
    image_size: int = 28
    in_channels: int = 1
    dataset: str = "mnist"

    # GAN
    latent_dim: int = 100
    disc_dropout: float = 0.3
    gan_lr: float = 2e-4
    gan_epochs: int = 5

    # Diffusion
    timesteps: int = 500
    base_channels: int = 32
    diff_lr: float = 1e-3
    diff_epochs: int = 5
    schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 0.02


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pixel_variance(images: torch.Tensor) -> float:
    """Mean per-image pixel variance — proxy for sample diversity."""
    return images.view(images.shape[0], -1).var(dim=1).mean().item()


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

class GANDiffusionComparison:
    """Trains GAN and DDPM on the same dataset and exposes comparison utilities."""

    def __init__(self):
        self.generator: Generator | None = None
        self.discriminator: Discriminator | None = None
        self.unet: UNet | None = None
        self.scheduler: NoiseScheduler | None = None

    # ------------------------------------------------------------------
    # GAN training
    # ------------------------------------------------------------------

    def train_gan(self, config: GANDiffusionConfig, logger: Logger) -> None:
        """Alternating D/G update loop — logs d_loss and g_loss per epoch."""
        G = Generator(
            latent_dim=config.latent_dim,
            image_size=config.image_size,
        ).to(config.device)
        D = Discriminator(
            image_size=config.image_size,
            dropout=config.disc_dropout,
        ).to(config.device)
        self.generator = G
        self.discriminator = D

        opt_d = optim.Adam(D.parameters(), lr=config.gan_lr, betas=(0.5, 0.999))
        opt_g = optim.Adam(G.parameters(), lr=config.gan_lr, betas=(0.5, 0.999))
        criterion = torch.nn.BCELoss()

        dl = get_dataloader(
            config.dataset, config.data_root, split="train",
            task="classification",
            batch_size=config.effective_batch_size,
            fast_demo=config.effective_fast_demo,
            sample_limit=config.dataset_sample_limit,
        )
        epochs = 1 if config.fast_demo else config.gan_epochs
        for epoch in range(epochs):
            G.train(); D.train()
            total_d = 0.0; total_g = 0.0
            for real_images, _ in dl:
                real_images = real_images.to(config.device)
                B = real_images.shape[0]

                # — Discriminator step —
                z = torch.randn(B, config.latent_dim, device=config.device)
                fake = G(z)
                d_loss = gan_d_loss(D, real_images, fake, criterion)
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()

                # — Generator step —
                z = torch.randn(B, config.latent_dim, device=config.device)
                fake = G(z)
                g_loss = gan_g_loss(D, fake, criterion)
                opt_g.zero_grad(); g_loss.backward(); opt_g.step()

                total_d += d_loss.item()
                total_g += g_loss.item()

            n = max(1, len(dl))
            logger.log_metrics(epoch, {
                "gan_d_loss": total_d / n,
                "gan_g_loss": total_g / n,
            })
            log.info(f"  [GAN]  epoch {epoch}  d={total_d/n:.4f}  g={total_g/n:.4f}")

        torch.save(G.state_dict(), logger.artifact_path("gan_generator.pt"))
        torch.save(D.state_dict(), logger.artifact_path("gan_discriminator.pt"))

    # ------------------------------------------------------------------
    # Diffusion training
    # ------------------------------------------------------------------

    def train_diffusion(self, config: GANDiffusionConfig, logger: Logger) -> None:
        """Standard DDPM training — logs mse loss per epoch."""
        from mini_networks.models.diffusion.scheduler import NoiseScheduler as NS

        unet = UNet(
            in_channels=config.in_channels,
            base_channels=config.base_channels,
        ).to(config.device)
        scheduler = NS(
            timesteps=config.effective_timesteps,
            schedule=config.schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        ).to(torch.device(config.device))
        self.unet = unet
        self.scheduler = scheduler

        optimizer = optim.Adam(unet.parameters(), lr=config.diff_lr)
        dl = get_dataloader(
            config.dataset, config.data_root, split="train",
            task="classification",
            batch_size=config.effective_batch_size,
            fast_demo=config.effective_fast_demo,
            sample_limit=config.dataset_sample_limit,
        )
        epochs = 1 if config.fast_demo else config.diff_epochs
        for epoch in range(epochs):
            unet.train()
            total = 0.0
            for images, _ in dl:
                images = images.to(config.device) * 2.0 - 1.0
                B = images.shape[0]
                t = torch.randint(0, config.effective_timesteps, (B,), device=config.device)
                noise = torch.randn_like(images)
                noisy = scheduler.add_noise(images, noise, t)
                pred = unet(noisy, t)
                loss = F.mse_loss(pred, noise)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total += loss.item()
            avg = total / max(1, len(dl))
            logger.log_metrics(epoch, {"diff_loss": avg})
            log.info(f"  [Diff] epoch {epoch}  loss {avg:.4f}")

        torch.save(unet.state_dict(), logger.artifact_path("diffusion.pt"))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def train_all(self, config: GANDiffusionConfig, logger: Logger) -> None:
        logger.log_config(config.model_dump())
        self.train_gan(config, logger)
        self.train_diffusion(config, logger)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_gan(
        self,
        config: GANDiffusionConfig,
        n_samples: int = 8,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Sample from the trained generator. Returns [n, 1, 28, 28] in [0, 1]."""
        assert self.generator is not None, "Train GAN first."
        G = self.generator
        G.eval()
        if seed is not None:
            torch.manual_seed(seed)
        z = torch.randn(n_samples, config.latent_dim, device=config.device)
        imgs = G(z)  # [n, 1, H, W], Tanh → [-1, 1]
        return ((imgs.clamp(-1, 1) + 1) / 2).cpu()

    @torch.no_grad()
    def sample_diffusion(
        self,
        config: GANDiffusionConfig,
        n_samples: int = 8,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Sample from the trained DDPM. Returns [n, 1, 28, 28] in [0, 1]."""
        assert self.unet is not None and self.scheduler is not None, "Train diffusion first."
        unet, scheduler = self.unet, self.scheduler
        unet.eval()
        x = sample_loop(
            scheduler=scheduler,
            predict_noise=lambda x, t_b, t, _: unet(x, t_b),
            shape=(n_samples, config.in_channels, config.image_size, config.image_size),
            device=config.device,
            timesteps=config.effective_timesteps,
            seed=seed,
        )
        return ((x.clamp(-1, 1) + 1) / 2).cpu()

    def compare(
        self,
        config: GANDiffusionConfig,
        n_samples: int = 8,
        seed: int | None = 42,
    ) -> dict[str, Any]:
        """Generate from both models and return a comparison dict.

        Returns:
            {
              "gan_samples":        Tensor [n, 1, 28, 28],
              "diffusion_samples":  Tensor [n, 1, 28, 28],
              "gan_diversity":      float (mean per-image pixel variance),
              "diffusion_diversity": float,
              "gan_mean_pixel":     float,
              "diffusion_mean_pixel": float,
            }
        """
        gan_imgs = self.sample_gan(config, n_samples=n_samples, seed=seed)
        diff_imgs = self.sample_diffusion(config, n_samples=n_samples, seed=seed)
        return {
            "gan_samples": gan_imgs,
            "diffusion_samples": diff_imgs,
            "gan_diversity": _pixel_variance(gan_imgs),
            "diffusion_diversity": _pixel_variance(diff_imgs),
            "gan_mean_pixel": gan_imgs.mean().item(),
            "diffusion_mean_pixel": diff_imgs.mean().item(),
        }
