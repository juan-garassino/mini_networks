from __future__ import annotations
from mini_networks.core.config import BaseConfig


class SimCLRConfig(BaseConfig):
    model_name: str = "simclr"
    proj_dim: int = 64
    temperature: float = 0.2
    dataset: str = "mnist"
