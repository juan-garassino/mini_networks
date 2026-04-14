"""UNet autoencoder trainer."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.unet_ae.config import UNetAEConfig
from mini_networks.models.unet_ae.model import UNetAutoencoder


class UNetAETrainer(BaseTrainer):
    def __init__(self):
        self.model: UNetAutoencoder | None = None

    def _build(self, config: UNetAEConfig) -> UNetAutoencoder:
        return UNetAutoencoder(base_channels=config.base_channels).to(config.device)

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, UNetAEConfig)
        model = self._build(config)
        self.model = model
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        logger.log_config(config.model_dump())

        for epoch in range(config.effective_epochs):
            model.train()
            total_loss = 0.0
            for images, _ in dataloader:
                images = images.to(config.device)
                recon = model(images)
                loss = F.mse_loss(recon, images)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.log_metrics(epoch, {
                "loss": total_loss / max(1, len(dataloader)),
                "epoch": epoch,
            })

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        assert isinstance(config, UNetAEConfig)
        if self.model is None:
            self.model = self._build(config)
        model = self.model
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(config.device)
                recon = model(images)
                total_loss += F.mse_loss(recon, images).item()
        return {"eval_loss": total_loss / max(1, len(dataloader))}

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, UNetAEConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        model = self.model
        model.eval()
        with torch.no_grad():
            images = inputs if not isinstance(inputs, dict) else inputs.get("images")
            images = images.to(config.device)
            recon = model(images)
        return {"recon": recon.cpu()}


def make_unet_ae_dataloader(config: UNetAEConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        task="classification",
        batch_size=config.effective_batch_size,
        fast_demo=config.effective_fast_demo,
        sample_limit=config.dataset_sample_limit,
    )
