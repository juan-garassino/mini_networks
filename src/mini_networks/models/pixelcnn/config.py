from __future__ import annotations
from mini_networks.core.config import BaseConfig


class PixelCNNConfig(BaseConfig):
    model_name: str = "pixelcnn"
    n_filters: int = 32
    n_layers: int = 4
    dataset: str = "mnist"
