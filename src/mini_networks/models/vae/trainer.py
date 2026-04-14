"""VAE trainer."""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.vae.config import VAEConfig
from mini_networks.models.vae.model import ConvVAE, vae_loss


class VAETrainer(BaseTrainer):
    def __init__(self):
        self.model: ConvVAE | None = None

    def _build(self, config: VAEConfig) -> ConvVAE:
        return ConvVAE(latent_dim=config.latent_dim, hidden_dim=config.hidden_dim).to(config.device)

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, VAEConfig)
        model = self._build(config)
        self.model = model
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        logger.log_config(config.model_dump())

        for epoch in range(config.effective_epochs):
            model.train()
            total = 0.0
            recon_total = 0.0
            kl_total = 0.0
            for images, _ in dataloader:
                images = images.to(config.device)
                recon, mu, logvar = model(images)
                loss, recon_loss, kl = vae_loss(recon, images, mu, logvar, beta=config.beta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
                recon_total += recon_loss.item()
                kl_total += kl.item()
            n = max(1, len(dataloader))
            logger.log_metrics(epoch, {
                "loss": total / n,
                "recon": recon_total / n,
                "kl": kl_total / n,
                "epoch": epoch,
            })

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        assert isinstance(config, VAEConfig)
        if self.model is None:
            self.model = self._build(config)
        model = self.model
        model.eval()
        total = 0.0
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(config.device)
                recon, mu, logvar = model(images)
                loss, _, _ = vae_loss(recon, images, mu, logvar, beta=config.beta)
                total += loss.item()
        return {"eval_loss": total / max(1, len(dataloader))}

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, VAEConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        model = self.model
        model.eval()
        with torch.no_grad():
            if isinstance(inputs, dict) and "sample" in inputs:
                n = int(inputs.get("sample", 1))
                z = torch.randn(n, config.latent_dim, device=config.device)
                samples = model.decode(z).cpu()
                return {"samples": samples}
            images = inputs if not isinstance(inputs, dict) else inputs.get("images")
            images = images.to(config.device)
            recon, _, _ = model(images)
            return {"recon": recon.cpu()}


def make_vae_dataloader(config: VAEConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        task="classification",
        batch_size=config.effective_batch_size,
        fast_demo=config.effective_fast_demo,
        sample_limit=config.dataset_sample_limit,
    )
