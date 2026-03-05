"""YOLO-style digit detector: shared CNN backbone → cls + bbox heads."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitDetector(nn.Module):
    """
    Dual-head CNN on 56x56 grayscale canvas.
    Outputs: (class_logits [B, 10], bbox_pred [B, 4]) where bbox is normalized [0,1].
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),  # 128 * 4 * 4 = 2048
        )
        self.cls_head = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        cls_logits = self.cls_head(features)
        bbox_pred = self.bbox_head(features)
        return cls_logits, bbox_pred


def detection_loss(
    cls_logits: torch.Tensor,
    bbox_pred: torch.Tensor,
    labels: torch.Tensor,
    bboxes: torch.Tensor,
    bbox_weight: float = 1.0,
) -> torch.Tensor:
    cls_loss = F.cross_entropy(cls_logits, labels)
    bbox_loss = F.mse_loss(bbox_pred, bboxes)
    return cls_loss + bbox_weight * bbox_loss
