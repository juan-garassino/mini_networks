from __future__ import annotations
from mini_networks.core.config import BaseConfig


class GANConfig(BaseConfig):
    model_name: str = "gan"
    latent_dim: int = 100
    image_size: int = 28
    in_channels: int = 1
    disc_dropout: float = 0.3
    lr: float = 0.0002
    dataset: str = "mnist"
