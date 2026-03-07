from __future__ import annotations
from mini_networks.core.config import BaseConfig


class VAEConfig(BaseConfig):
    model_name: str = "vae"
    latent_dim: int = 32
    hidden_dim: int = 128
    beta: float = 1.0
    dataset: str = "mnist"
