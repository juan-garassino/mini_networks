from __future__ import annotations
from mini_networks.core.config import BaseConfig


class PixelCNNConfig(BaseConfig):
    model_name: str = "pixelcnn"
    # 8 layers / 64 filters (was 4/32): the receptive field of 4 stacked 3x3
    # masked convs (~9px) physically cannot coordinate a 28px digit — samples
    # were local stroke fragments at any budget (m-vision-7).
    n_filters: int = 64
    n_layers: int = 8
    dataset: str = "mnist"
