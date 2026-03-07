from __future__ import annotations
from mini_networks.core.config import BaseConfig


class TabularDiffusionConfig(BaseConfig):
    model_name: str = "tabular_diffusion"
    n_features: int = 4
    timesteps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02
    hidden_dim: int = 128
    dataset: str = "iris"
    require_downloads: bool = True
