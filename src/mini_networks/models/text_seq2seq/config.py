from __future__ import annotations
from mini_networks.core.config import BaseConfig


class TextSeq2SeqConfig(BaseConfig):
    model_name: str = "text_seq2seq"
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 128
    seq_len: int = 64
    vocab_size: int = 256
    dataset: str = "text_file"
    text_file: str = ""
