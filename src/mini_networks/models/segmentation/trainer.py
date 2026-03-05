"""Segmentation trainer (binary + multiclass UNet)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.segmentation.config import SegmentationConfig
from mini_networks.models.segmentation.unet import SegUNet, dice_loss, multiclass_dice_loss


class SegmentationTrainer(BaseTrainer):
    def __init__(self):
        self.model: SegUNet | None = None

    def _build(self, config: SegmentationConfig) -> SegUNet:
        return SegUNet(
            in_channels=1,
            out_channels=config.out_channels,
            base_channels=config.base_channels,
        ).to(config.device)

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, SegmentationConfig)
        model = self._build(config)
        self.model = model
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        logger.log_config(config.model_dump())

        for epoch in range(config.effective_epochs):
            model.train()
            total_loss = 0.0
            for images, masks in dataloader:
                images = images.to(config.device)
                masks = masks.to(config.device)
                preds = model(images)
                if config.task_mode == "binary":
                    loss = (
                        nn.BCELoss()(preds.squeeze(1), masks.float())
                        + dice_loss(preds.squeeze(1), masks)
                    )
                else:
                    loss = (
                        F.cross_entropy(preds, masks)
                        + multiclass_dice_loss(preds, masks, config.num_classes)
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / max(1, len(dataloader))
            logger.log_metrics(epoch, {"loss": avg, "epoch": epoch})

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        assert isinstance(config, SegmentationConfig)
        if self.model is None:
            self.model = self._build(config)
        model = self.model
        model.eval()
        total_iou = 0.0
        n = 0
        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(config.device)
                masks = masks.to(config.device)
                preds = model(images)
                if config.task_mode == "binary":
                    pred_bin = (preds.squeeze(1) > 0.5).long()
                    intersection = (pred_bin & masks.bool()).float().sum()
                    union = (pred_bin | masks.bool()).float().sum()
                    total_iou += (intersection / (union + 1e-6)).item()
                n += 1
        return {"eval_iou": total_iou / max(1, n)}

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, SegmentationConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        self.model.eval()
        with torch.no_grad():
            images = inputs if isinstance(inputs, torch.Tensor) else inputs["images"]
            images = images.to(config.device)
            return {"masks": self.model(images).cpu()}


def make_segmentation_dataloader(config: SegmentationConfig, split: str = "train") -> DataLoader:
    task = "binary_segmentation" if config.task_mode == "binary" else "multiclass_segmentation"
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        task=task,
        batch_size=config.effective_batch_size,
        fast_demo=config.fast_demo,
    )
