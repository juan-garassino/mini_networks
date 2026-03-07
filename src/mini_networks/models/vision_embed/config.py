from __future__ import annotations
from mini_networks.core.config import BaseConfig


class VisionEmbedConfig(BaseConfig):
    model_name: str = "vision_embed"
    embed_dim: int = 64
    temperature: float = 0.2
    dataset: str = "mnist"
