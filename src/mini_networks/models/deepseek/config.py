"""Config for the mini DeepSeek V4 language model."""
from __future__ import annotations

from mini_networks.models.transformer.config import TransformerConfig


class DeepseekConfig(TransformerConfig):
    """Mini DeepSeek V4 (arXiv 2606.19348) as a first-class zoo entry.

    Same char-level corpus, tokenizer, and trainer contract as `transformer` /
    `moe` / `kimi`, so the eval_loss column is a direct architecture
    comparison. The V4-specific machinery (CSA/HCA compressed attention, mHC
    hyper-connections, DeepSeekMoE) lives in models/deepseek/model.py.
    """

    model_name: str = "deepseek"

    n_layers: int = 8          # interleaved CSA/HCA (see hca_every)

    # --- Hybrid compressed attention ---
    csa_m: int = 4             # CSA: compress every m tokens into one KV entry
    csa_top_k: int = 8         # CSA: compressed entries each query attends to (sparse)
    hca_m: int = 16            # HCA: heavier compression, dense attention over entries
    hca_every: int = 4         # every hca_every-th layer is HCA (3 CSA : 1 HCA — mini choice)
    kv_dim: int = 64           # compressed KV entry dim c (queries live in this space too)
    n_win: int = 16            # sliding-window branch: raw KV entries for the last n_win tokens
    rope_dims: int = 16        # partial RoPE: rotate only the last rope_dims of q / entries / outputs

    # --- Manifold-Constrained Hyper-Connections ---
    use_mhc: bool = True       # ablation flag: False -> plain sequential residual stream
    n_hc: int = 2              # residual stream width multiplier (paper-typical small n)
    sinkhorn_iters: int = 10   # Sinkhorn-Knopp iterations projecting B onto doubly stochastic

    # --- DeepSeekMoE FFN ---
    ds_num_experts: int = 8    # fine-grained routed expert pool
    ds_top_k: int = 2          # active routed experts per token
    ds_num_shared: int = 1     # always-on shared experts
    balance_gamma: float = 0.01  # aux-loss-free bias-balancing step (sign rule)

    # --- Multi-Token Prediction ---
    mtp_weight: float = 0.1    # weight of the t+2 prediction CE folded into aux_loss
