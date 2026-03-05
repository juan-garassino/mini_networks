from __future__ import annotations
from typing import Literal
from mini_networks.core.config import BaseConfig


class TransformerConfig(BaseConfig):
    model_name: str = "transformer"
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 256
    seq_len: int = 128
    vocab_size: int = 256
    dropout: float = 0.1
    dataset: str = "text_file"
    text_file: str = ""

    # FFN block type: "standard" | "moe" | "mamba"
    block_type: Literal["standard", "moe", "mamba"] = "standard"

    # MoE hyperparameters (used when block_type == "moe")
    moe_num_experts: int = 4
    moe_top_k: int = 1
    moe_router_hidden: int = 64
    moe_balance_loss_weight: float = 0.02
    moe_entropy_bonus: float = 0.001
    moe_router_temp: float = 1.0
    moe_add_gumbel: bool = True
    moe_shared_scale: float = 0.3

    # Mamba hyperparameters (used when block_type == "mamba")
    mamba_d_state: int = 16
    mamba_d_conv: int = 4

    # Tokenizer: "char" (default) or "bpe"
    tokenizer_type: Literal["char", "bpe"] = "char"
    bpe_vocab_size: int = 512  # target vocab size when tokenizer_type == "bpe"
