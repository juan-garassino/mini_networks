from __future__ import annotations
from mini_networks.core.config import BaseConfig


class CLIPConfig(BaseConfig):
    model_name: str = "clip"
    embed_dim: int = 128
    image_size: int = 28
    patch_size: int = 4
    vocab_size: int = 256
    text_seq_len: int = 32
    text_d_model: int = 64
    text_n_heads: int = 2
    text_n_layers: int = 2
    temperature: float = 0.07
    dataset: str = "mnist"
