from __future__ import annotations
from mini_networks.core.config import BaseConfig


class TextTokenClassifierConfig(BaseConfig):
    model_name: str = "text_token_classifier"
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    seq_len: int = 64
    vocab_size: int = 256
    dataset: str = "text_file"
    text_file: str = ""
