"""MobileNet trainer."""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import SupervisedTrainer
from mini_networks.models.mobilenet.config import MobileNetConfig
from mini_networks.models.mobilenet.model import TinyMobileNet


class MobileNetTrainer(SupervisedTrainer):
    def __init__(self):
        self.model: TinyMobileNet | None = None

    def _build(self, config: MobileNetConfig) -> TinyMobileNet:
        return TinyMobileNet(num_classes=config.num_classes, width_mult=config.width_mult).to(config.device)

    def _forward(self, model, batch, config: MobileNetConfig):
        images, labels = batch
        images = images.to(config.device)
        return model(images), labels

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, MobileNetConfig)
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


def make_mobilenet_dataloader(config: MobileNetConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        task="classification",
        batch_size=config.effective_batch_size,
        fast_demo=config.effective_fast_demo,
        sample_limit=config.dataset_sample_limit,
    )
