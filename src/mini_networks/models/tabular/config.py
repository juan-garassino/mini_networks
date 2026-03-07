from __future__ import annotations
from typing import Literal
from mini_networks.core.config import BaseConfig


class TabularClassifierConfig(BaseConfig):
    model_name: str = "tabular_classifier"
    n_features: int = 4
    n_classes: int = 3
    model_type: Literal["mlp", "linear", "transformer"] = "mlp"
    hidden_dim: int = 64
    dataset: str = "iris"
    require_downloads: bool = True
