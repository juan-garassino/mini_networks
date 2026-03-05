from __future__ import annotations
from mini_networks.core.config import BaseConfig


class RAGConfig(BaseConfig):
    model_name: str = "rag"

    # Retrieval
    top_k: int = 3              # number of documents to retrieve
    chunk_size: int = 200       # characters per document chunk

    # Generation (uses TransformerLM underneath)
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    d_ff: int = 128
    seq_len: int = 128
    vocab_size: int = 256
    dropout: float = 0.1

    dataset: str = "text_file"
    text_file: str = ""
