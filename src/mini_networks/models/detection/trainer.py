"""Detection trainer (dual-head CNN: classification + bounding box)."""
from __future__ import annotations

from typing import Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.detection.config import DetectionConfig
from mini_networks.models.detection.model import DigitDetector, detection_loss


class DetectionTrainer(BaseTrainer):
    def __init__(self):
        self.model: DigitDetector | None = None

    def _build(self, config: DetectionConfig) -> DigitDetector:
        return DigitDetector(num_classes=config.num_classes).to(config.device)

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, DetectionConfig)
        model = self._build(config)
        self.model = model
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        logger.log_config(config.model_dump())

        for epoch in range(config.effective_epochs):
            model.train()
            total_loss = 0.0
            for canvases, labels, bboxes in dataloader:
                canvases = canvases.to(config.device)
                labels = labels.to(config.device)
                bboxes = bboxes.to(config.device)
                cls_logits, bbox_pred = model(canvases)
                loss = detection_loss(
                    cls_logits, bbox_pred, labels, bboxes, config.bbox_loss_weight
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / max(1, len(dataloader))
            logger.log_metrics(epoch, {"loss": avg, "epoch": epoch})

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        assert isinstance(config, DetectionConfig)
        if self.model is None:
            self.model = self._build(config)
        model = self.model
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for canvases, labels, _ in dataloader:
                canvases = canvases.to(config.device)
                labels = labels.to(config.device)
                cls_logits, _ = model(canvases)
                preds = cls_logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        return {"eval_accuracy": correct / max(1, total)}

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, DetectionConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        self.model.eval()
        with torch.no_grad():
            images = inputs if isinstance(inputs, torch.Tensor) else inputs["images"]
            images = images.to(config.device)
            cls_logits, bbox_pred = self.model(images)
            return {
                "class_probs": torch.softmax(cls_logits, dim=-1).cpu(),
                "bboxes": bbox_pred.cpu(),
            }


def make_detection_dataloader(config: DetectionConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        task="detection",
        batch_size=config.effective_batch_size,
        fast_demo=config.fast_demo,
        canvas_size=config.canvas_size,
    )
