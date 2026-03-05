from __future__ import annotations
from typing import Literal
from mini_networks.core.config import BaseConfig


class RNNConfig(BaseConfig):
    model_name: str = "rnn"
    cell_type: Literal["rnn", "lstm", "gru"] = "lstm"
    hidden_dim: int = 256
    n_layers: int = 2
    seq_len: int = 128
    vocab_size: int = 256
    dropout: float = 0.1
    dataset: str = "text_file"
    text_file: str = ""
