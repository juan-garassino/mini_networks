"""SimCLR trainer."""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.runtime import ContrastiveTrainer
from mini_networks.models.simclr.config import SimCLRConfig
from mini_networks.models.simclr.model import SimCLREncoder, info_nce_loss


class SimCLRTrainer(ContrastiveTrainer):
    def __init__(self):
        self.model: SimCLREncoder | None = None

    def _build(self, config: SimCLRConfig) -> SimCLREncoder:
        return SimCLREncoder(proj_dim=config.proj_dim).to(config.device)

    def _forward(self, model: SimCLREncoder, batch, config: SimCLRConfig):
        v1, v2, _ = batch
        v1, v2 = v1.to(config.device), v2.to(config.device)
        z1 = model(v1)
        z2 = model(v2)
        return z1, z2

    def _loss(self, emb_a: torch.Tensor, emb_b: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        return info_nce_loss(emb_a, emb_b, temperature=temperature)

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, SimCLRConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        model = self.model
        model.eval()
        with torch.no_grad():
            images = inputs if not isinstance(inputs, dict) else inputs.get("images")
            images = images.to(config.device)
            embeds = model(images)
        return {"embeddings": embeds.cpu()}


def make_simclr_dataloader(config: SimCLRConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        task="contrastive",
        batch_size=config.effective_batch_size,
        fast_demo=config.effective_fast_demo,
        sample_limit=config.dataset_sample_limit,
        image_size=28,
    )
