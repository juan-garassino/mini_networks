"""Vision embedding trainer (contrastive)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.runtime import ContrastiveTrainer
from mini_networks.models.vision_embed.config import VisionEmbedConfig
from mini_networks.models.vision_embed.model import VisionEmbedCNN


class VisionEmbedTrainer(ContrastiveTrainer):
    def __init__(self):
        self.model: VisionEmbedCNN | None = None

    def _build(self, config: VisionEmbedConfig) -> VisionEmbedCNN:
        return VisionEmbedCNN(embed_dim=config.embed_dim).to(config.device)

    def _forward(self, model: VisionEmbedCNN, batch, config: VisionEmbedConfig):
        v1, v2, _ = batch
        v1, v2 = v1.to(config.device), v2.to(config.device)
        z1 = model(v1)
        z2 = model(v2)
        return z1, z2

    def _loss(self, emb_a: torch.Tensor, emb_b: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        logits = (emb_a @ emb_b.T) / temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, targets)

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, VisionEmbedConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        model = self.model
        model.eval()
        with torch.no_grad():
            images = inputs if not isinstance(inputs, dict) else inputs.get("images")
            images = images.to(config.device)
            emb = model(images)
        return {"embeddings": emb.cpu()}


def make_vision_embed_dataloader(config: VisionEmbedConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        task="contrastive",
        batch_size=config.effective_batch_size,
        fast_demo=config.fast_demo,
    )
