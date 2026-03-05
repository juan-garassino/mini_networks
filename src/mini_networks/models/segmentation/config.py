from __future__ import annotations
from typing import Literal
from mini_networks.core.config import BaseConfig


class SegmentationConfig(BaseConfig):
    model_name: str = "segmentation"
    task_mode: Literal["binary", "multiclass"] = "binary"
    num_classes: int = 1
    input_size: int = 28
    base_channels: int = 32
    dataset: str = "mnist"

    @property
    def out_channels(self) -> int:
        return 1 if self.task_mode == "binary" else self.num_classes
