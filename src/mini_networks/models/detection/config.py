from __future__ import annotations
from mini_networks.core.config import BaseConfig


class DetectionConfig(BaseConfig):
    model_name: str = "detection"
    canvas_size: int = 56
    num_classes: int = 10
    bbox_loss_weight: float = 1.0
    dataset: str = "mnist"
