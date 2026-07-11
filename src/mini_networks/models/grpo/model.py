"""GRPO: Group Relative Policy Optimization (DeepSeekMath, Shao et al. 2024).

The idea in one line: PPO needs a learned VALUE network to estimate the
baseline that turns rewards into advantages; GRPO deletes it. Instead, sample
a GROUP of G responses for the SAME prompt and use the group's own statistics
as the baseline:

    A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)

Every response is judged relative to its siblings — "better than the other
answers to this question" — which is exactly the signal a preference-style
reward provides, with zero extra parameters and no value-loss tuning. The
clipped-ratio surrogate and the KL penalty against a frozen reference model
are unchanged from PPO.

This mini version reuses the rlhf pipeline wholesale (same TransformerLM,
same Shakespeare corpus, same dense heuristic reward): pretrain → freeze
reference → iterate [sample G responses per prompt → group-normalize rewards
→ clipped update + KL]. Contrast it with `rlhf` (PPO) trained on the same
data — that pairing is the lesson.

Deliberately simplified: token-level advantages are the response's scalar
advantage broadcast to every token (no per-token credit), one prompt-group
per update batch, and the same tiny char-LM as every other text item.
"""
from __future__ import annotations

from mini_networks.models.rlhf.model import RewardModel, shakespearean_score  # noqa: F401
from mini_networks.models.transformer.model import TransformerLM  # noqa: F401
