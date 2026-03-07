from __future__ import annotations
from mini_networks.core.config import BaseConfig


class ViTConfig(BaseConfig):
    model_name: str = "vit"
    patch_size: int = 4
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    mlp_dim: int = 128
    num_classes: int = 10
    dataset: str = "mnist"
