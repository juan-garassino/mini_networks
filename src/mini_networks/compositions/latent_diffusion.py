"""Latent diffusion composition: VAE + UNet on latents."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from mini_networks.core.logging.logger import Logger
from mini_networks.core.data.registry import get_dataloader
from mini_networks.models.diffusion.vae import VAE, vae_loss
from mini_networks.models.diffusion.model import UNet
from mini_networks.models.diffusion.scheduler import NoiseScheduler
from mini_networks.core.config import BaseConfig
from mini_networks.core.diffusion.sampling import sample_loop


class LatentDiffusionConfig(BaseConfig):
    model_name: str = "latent_diffusion"
    latent_channels: int = 4
    timesteps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02
    dataset: str = "mnist"


class LatentDiffusion:
    def __init__(self):
        self.vae: VAE | None = None
        self.unet: UNet | None = None
        self.scheduler: NoiseScheduler | None = None

    def train(self, config: LatentDiffusionConfig, logger: Logger) -> None:
        dl = get_dataloader(
            name=config.dataset,
            data_root=config.data_root,
            split="train",
            task="classification",
            batch_size=config.effective_batch_size,
            fast_demo=config.effective_fast_demo,
            sample_limit=config.dataset_sample_limit,
        )
        vae = VAE(latent_channels=config.latent_channels).to(config.device)
        unet = UNet(in_channels=config.latent_channels, base_channels=32).to(config.device)
        scheduler = NoiseScheduler(
            timesteps=config.timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        ).to(torch.device(config.device))
        self.vae, self.unet, self.scheduler = vae, unet, scheduler

        opt_vae = torch.optim.Adam(vae.parameters(), lr=config.learning_rate)
        opt_unet = torch.optim.Adam(unet.parameters(), lr=config.learning_rate)

        # Train VAE briefly
        for epoch in range(config.effective_epochs):
            vae.train()
            total = 0.0
            for images, _ in dl:
                images = images.to(config.device) * 2.0 - 1.0
                recon, mu, logvar = vae(images)
                loss = vae_loss(recon, images, mu, logvar)
                opt_vae.zero_grad()
                loss.backward()
                opt_vae.step()
                total += loss.item()
            logger.log_metrics(epoch, {"vae_loss": total / max(1, len(dl))})

        # Train UNet in latent space — VAE must be frozen in eval mode so its
        # BatchNorm stats and reparameterisation noise don't shift the latents
        vae.eval()
        for epoch in range(config.effective_epochs):
            unet.train()
            total = 0.0
            for images, _ in dl:
                images = images.to(config.device) * 2.0 - 1.0
                with torch.no_grad():
                    mu, logvar = vae.encode(images)
                    z = vae.reparameterise(mu, logvar)
                B = z.size(0)
                t = torch.randint(0, config.timesteps, (B,), device=config.device)
                noise = torch.randn_like(z)
                zt = scheduler.add_noise(z, noise, t)
                pred = unet(zt, t)
                loss = F.mse_loss(pred, noise)
                opt_unet.zero_grad()
                loss.backward()
                opt_unet.step()
                total += loss.item()
            logger.log_metrics(epoch, {"latent_loss": total / max(1, len(dl))})

        torch.save(vae.state_dict(), logger.artifact_path("vae.pt"))
        torch.save(unet.state_dict(), logger.artifact_path("unet.pt"))

    @torch.no_grad()
    def sample(self, config: LatentDiffusionConfig, n: int = 4):
        if self.vae is None or self.unet is None or self.scheduler is None:
            raise RuntimeError("Train first.")
        self.vae.eval()
        self.unet.eval()
        z = sample_loop(
            scheduler=self.scheduler,
            predict_noise=lambda z, t_batch, t, _: self.unet(z, t_batch),
            shape=(n, config.latent_channels, 7, 7),
            device=config.device,
            timesteps=config.timesteps,
        )
        images = self.vae.decode(z)
        return (images + 1.0) / 2.0
