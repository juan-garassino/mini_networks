"""GAN trainer: alternating Discriminator / Generator updates per batch."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.gan.config import GANConfig
from mini_networks.models.gan.model import Discriminator, Generator, gan_d_loss, gan_g_loss

import logging

log = logging.getLogger(__name__)


class GANTrainer(BaseTrainer):
    def __init__(self):
        self.generator: Generator | None = None
        self.discriminator: Discriminator | None = None

    def _build(self, config: GANConfig) -> tuple[Generator, Discriminator]:
        G = Generator(
            latent_dim=config.latent_dim,
            image_size=config.image_size,
            in_channels=config.in_channels,
        ).to(config.device)
        D = Discriminator(
            image_size=config.image_size,
            in_channels=config.in_channels,
            dropout=config.disc_dropout,
        ).to(config.device)
        return G, D

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, GANConfig)
        G, D = self._build(config)
        self.generator = G
        self.discriminator = D

        criterion = nn.BCELoss()
        opt_g = optim.Adam(G.parameters(), lr=config.lr, betas=(0.5, 0.999))
        opt_d = optim.Adam(D.parameters(), lr=config.lr, betas=(0.5, 0.999))
        logger.log_config(config.model_dump())

        for epoch in range(config.effective_epochs):
            G.train(); D.train()
            total_d, total_g = 0.0, 0.0

            for batch_idx, batch in enumerate(dataloader):
                if config.max_train_batches is not None and batch_idx >= config.max_train_batches:
                    break
                # dataloader may return (image, label) or just image
                real = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(config.device)
                B = real.size(0)
                z = torch.randn(B, config.latent_dim, device=config.device)
                fake = G(z)

                # --- Discriminator step ---
                opt_d.zero_grad()
                d_loss = gan_d_loss(D, real, fake, criterion)
                d_loss.backward()
                opt_d.step()

                # --- Generator step (fresh fake to avoid stale graph) ---
                z2 = torch.randn(B, config.latent_dim, device=config.device)
                fake2 = G(z2)
                opt_g.zero_grad()
                g_loss = gan_g_loss(D, fake2, criterion)
                g_loss.backward()
                opt_g.step()

                total_d += d_loss.item()
                total_g += g_loss.item()

            n = max(1, len(dataloader))
            avg_d = total_d / n
            avg_g = total_g / n
            logger.log_metrics(epoch, {"d_loss": avg_d, "g_loss": avg_g, "epoch": epoch})
            log.info(f"  epoch {epoch}  d_loss {avg_d:.4f}  g_loss {avg_g:.4f}")

        torch.save(G.state_dict(), logger.artifact_path("generator.pt"))
        torch.save(D.state_dict(), logger.artifact_path("discriminator.pt"))

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        """Returns mean discriminator score on real images (should be ~0.5 after training)."""
        assert isinstance(config, GANConfig)
        if self.discriminator is None:
            _, self.discriminator = self._build(config)
        D = self.discriminator
        D.eval()
        total_score = 0.0
        n = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if config.max_eval_batches is not None and batch_idx >= config.max_eval_batches:
                    break
                real = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(config.device)
                scores = D(real)
                total_score += scores.mean().item()
                n += 1
        return {"mean_real_score": total_score / max(1, n)}

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        """Generate samples. inputs: {"n_samples": 8, "seed": 42}."""
        assert isinstance(config, GANConfig)
        if self.generator is None:
            raise RuntimeError("Generator not loaded. Call train() first.")
        G = self.generator
        G.eval()
        n_samples = inputs.get("n_samples", 8) if isinstance(inputs, dict) else 8
        seed = inputs.get("seed") if isinstance(inputs, dict) else None
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            z = torch.randn(n_samples, config.latent_dim, device=config.device)
            samples = G(z).cpu()
        # Scale from [-1, 1] (Tanh) to [0, 1]
        samples = (samples + 1) / 2
        return {"samples": samples}


    def load_checkpoint(self, config: BaseConfig, artifacts_dir) -> None:
        """Load generator.pt + discriminator.pt from artifacts_dir."""
        from pathlib import Path
        assert isinstance(config, GANConfig)
        path = Path(artifacts_dir)
        G, D = self._build(config)
        G.load_state_dict(torch.load(path / "generator.pt", map_location=config.device, weights_only=True))
        D.load_state_dict(torch.load(path / "discriminator.pt", map_location=config.device, weights_only=True))
        G.eval()
        D.eval()
        self.generator = G
        self.discriminator = D


def make_gan_dataloader(config: GANConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        task="classification",
        fast_demo=config.effective_fast_demo,
        sample_limit=config.dataset_sample_limit,
        batch_size=config.effective_batch_size,
    )
