"""ViT trainer."""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import SupervisedTrainer
from mini_networks.models.vit.config import ViTConfig
from mini_networks.models.vit.model import MiniViT


class ViTTrainer(SupervisedTrainer):
    def __init__(self):
        self.model: MiniViT | None = None

    def _build(self, config: ViTConfig) -> MiniViT:
        return MiniViT(
            patch_size=config.patch_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            mlp_dim=config.mlp_dim,
            num_classes=config.num_classes,
        ).to(config.device)

    def _forward(self, model, batch, config: ViTConfig):
        images, labels = batch
        images = images.to(config.device)
        return model(images), labels

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, ViTConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        model = self.model
        model.eval()
        with torch.no_grad():
            images = inputs if not isinstance(inputs, dict) else inputs.get("images")
            images = images.to(config.device)
            logits = model(images)
            preds = logits.argmax(dim=-1)
        return {"predictions": preds.cpu().tolist(), "logits": logits.cpu().tolist()}


def make_vit_dataloader(config: ViTConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        task="classification",
        batch_size=config.effective_batch_size,
        fast_demo=config.fast_demo,
    )
