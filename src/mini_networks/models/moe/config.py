from __future__ import annotations

from typing import Literal

from mini_networks.models.transformer.config import TransformerConfig


class MoEConfig(TransformerConfig):
    """Mixture-of-Experts LM as a first-class zoo entry.

    The MoE machinery (router, top-k dispatch, load-balancing loss, optional
    Gumbel noise) already lives inside TransformerLM behind
    ``block_type="moe"`` — this config surfaces it as its own model so the
    zoo teaches it explicitly, and so `transformer` vs `moe` on the same
    corpus is a direct dense-vs-sparse comparison.
    """

    model_name: str = "moe"
    block_type: Literal["standard", "moe", "mamba"] = "moe"
