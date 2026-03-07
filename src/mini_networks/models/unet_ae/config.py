from __future__ import annotations
from mini_networks.core.config import BaseConfig


class UNetAEConfig(BaseConfig):
    model_name: str = "unet_ae"
    base_channels: int = 32
    dataset: str = "mnist"
