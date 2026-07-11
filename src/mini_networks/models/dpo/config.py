from __future__ import annotations

from mini_networks.models.rlhf.config import RLHFConfig


class DPOConfig(RLHFConfig):
    model_name: str = "dpo"
    # Inverse-temperature of the implicit reward. Small beta = gentle
    # preference pressure (paper sweeps 0.1-0.5).
    dpo_beta: float = 0.1
