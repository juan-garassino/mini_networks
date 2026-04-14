"""Segmentation trainer (binary + multiclass UNet)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import SegmentationTrainerBase
from mini_networks.models.segmentation.config import SegmentationConfig
from mini_networks.models.segmentation.unet import SegUNet, dice_loss, multiclass_dice_loss


class SegmentationTrainer(SegmentationTrainerBase):
    def __init__(self):
        self.model: SegUNet | None = None

    def _build(self, config: SegmentationConfig) -> SegUNet:
        return SegUNet(
            in_channels=1,
            out_channels=config.out_channels,
            base_channels=config.base_channels,
        ).to(config.device)

    def _forward(self, model: SegUNet, batch, config: SegmentationConfig):
        images, masks = batch
        images = images.to(config.device)
        masks = masks.to(config.device)
        preds = model(images)
        return preds, masks

    def _loss(self, preds: torch.Tensor, targets: torch.Tensor, config: BaseConfig) -> torch.Tensor:
        assert isinstance(config, SegmentationConfig)
        if config.task_mode == "binary":
            return nn.BCELoss()(preds.squeeze(1), targets.float()) + dice_loss(
                preds.squeeze(1), targets
            )
        return F.cross_entropy(preds, targets) + multiclass_dice_loss(
            preds, targets, config.num_classes
        )

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
        fast_demo=config.effective_fast_demo,
        sample_limit=config.dataset_sample_limit,
    )
