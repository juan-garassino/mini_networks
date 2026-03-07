from __future__ import annotations
from mini_networks.core.config import BaseConfig


class ClassifierConfig(BaseConfig):
    model_name: str = "classifier"
    hidden_dim: int = 64
    num_classes: int = 10
    dataset: str = "mnist"
