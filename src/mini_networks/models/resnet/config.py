from __future__ import annotations
from mini_networks.core.config import BaseConfig


class ResNetConfig(BaseConfig):
    model_name: str = "resnet"
    base_channels: int = 32
    num_classes: int = 10
    dataset: str = "mnist"
