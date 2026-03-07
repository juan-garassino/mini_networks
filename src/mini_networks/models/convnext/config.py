from __future__ import annotations
from mini_networks.core.config import BaseConfig


class ConvNeXtConfig(BaseConfig):
    model_name: str = "convnext"
    base_channels: int = 32
    num_classes: int = 10
    dataset: str = "mnist"
