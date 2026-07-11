from __future__ import annotations

from mini_networks.models.rlhf.config import RLHFConfig


class GRPOConfig(RLHFConfig):
    model_name: str = "grpo"
    # Responses sampled PER PROMPT — the group whose mean reward is the
    # baseline. The paper uses 64; at mini scale 4-8 already shows the effect.
    group_size: int = 4
