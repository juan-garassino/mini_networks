"""YOLO-style digit detector: shared CNN backbone -> cls + bbox heads.

Object detection as direct regression (the single-shot idea behind YOLO):
one CNN forward pass predicts both what the object is and where it is — no
region proposals, no sliding windows. A shared backbone extracts features
once; two small heads branch off it, one for class logits and one for box
coordinates, trained jointly with a summed loss:

    L = CrossEntropy(cls_logits, label) + w * MSE(bbox_pred, bbox)

This implementation, on a 56x56 canvas containing one MNIST digit:

    [B,1,56,56] -> Conv3x3(1->32) +ReLU -> MaxPool2     28x28
                -> Conv3x3(32->64) +ReLU -> MaxPool2    14x14
                -> Conv3x3(64->128)+ReLU -> AdaptiveAvgPool(4) -> 4x4
                -> Flatten(2048)
       cls head:  Linear(2048->256) -> ReLU -> Linear(256->10)
       bbox head: Linear(2048->256) -> ReLU -> Linear(256->4) -> Sigmoid

The sigmoid keeps box predictions in [0,1], i.e. coordinates normalised to
canvas size, so the MSE box loss is scale-free.

Deliberately simplified: exactly one object per image, so there is no grid
of cells, no anchor boxes, no objectness score, and no non-max suppression —
the parts of real YOLO that handle multiple and overlapping detections.
MSE on raw coordinates also stands in for IoU-based box losses.
"""
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
