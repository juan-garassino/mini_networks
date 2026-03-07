from __future__ import annotations
from mini_networks.core.config import BaseConfig


class MobileNetConfig(BaseConfig):
    model_name: str = "mobilenet"
    width_mult: float = 1.0
    num_classes: int = 10
    dataset: str = "mnist"
