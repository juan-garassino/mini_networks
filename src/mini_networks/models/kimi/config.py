"""Config for the mini Kimi K3 language model."""
from __future__ import annotations

from typing import Literal

from mini_networks.models.transformer.config import TransformerConfig


class KimiConfig(TransformerConfig):
    """Mini Kimi K3 (arXiv 2607.24653) as a first-class zoo entry.

    Same char-level corpus, tokenizer, and trainer contract as `transformer` /
    `moe`, so the eval_loss column is a direct architecture comparison. The
    K3-specific machinery (KDA hybrid attention, Block AttnRes, Stable
    LatentMoE) lives in models/kimi/model.py.
    """

    model_name: str = "kimi"
    block_type: Literal["standard", "moe", "mamba"] = "standard"  # unused by KimiLM

    n_layers: int = 8          # 2 hybrid blocks of 4 (+1 final global layer added by the model)

    # --- KDA (Kimi Delta Attention) ---
    kda_per_block: int = 3     # KDA layers per hybrid block; the 4th is gated full attention (paper 3:1)
    kda_g_min: float = -5.0    # lower bound on log-decay: alpha in (e^-5, 1) (paper Eq. 5)
    kda_conv_kernel: int = 3   # causal depthwise ShortConv kernel on q/k/v projections
    kda_decay_rank: int = 16   # low-rank bottleneck for the per-channel decay logits

    # --- Block Attention Residuals ---
    use_attn_res: bool = True      # ablation flag: False -> plain sequential residual stream
    attn_res_block_size: int = 4   # layers summed into one block representation (paper S=L/N)

    # --- Stable LatentMoE FFN (one per attention layer) ---
    latent_dim: int = 64       # routed experts operate in this latent space (W_down: d_model->latent)
    num_routed: int = 8        # routed expert pool (paper: 896; mini scale)
    router_top_k: int = 2      # active routed experts per token (paper: 16)
    num_shared: int = 2        # always-on full-width experts (paper Table 1: 2)
    situ_beta1: float = 4.0    # SiTU-GLU gate-branch softcap (paper Eq. 12)
    situ_beta2: float = 25.0   # SiTU-GLU up-branch softcap
    balance_gamma: float = 0.01  # sign-rule bias-balancing step (stands in for Quantile Balancing)

    # --- Multi-Token Prediction (paper Table 1: 1 MTP layer) ---
    mtp_weight: float = 0.1    # weight of the t+2 prediction CE folded into aux_loss
