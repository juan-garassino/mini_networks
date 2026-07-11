"""DPO: Direct Preference Optimization (Rafailov et al., 2023).

The idea in one line: RLHF's whole PPO loop (rollouts, value baselines,
clipped ratios, KL controllers) can be replaced by a single supervised loss
on preference PAIRS. For a chosen response y+ and rejected response y- to the
same prompt, DPO maximizes

    log sigmoid( beta * [ (log pi(y+) - log ref(y+))
                        - (log pi(y-) - log ref(y-)) ] )

i.e. push the policy's log-prob margin over the frozen reference up for the
preferred response and down for the rejected one. The KL anchor is IMPLICIT
in the ref-model terms — no reward model, no sampling during optimization,
no RL machinery at all.

This mini version reuses the rlhf pipeline (same TransformerLM, corpus, and
dense heuristic score): pretrain → freeze reference → build preference pairs
by sampling two responses per prompt and letting the heuristic score pick
chosen vs rejected → DPO updates. Trained beside `rlhf` (PPO) and `grpo`,
the three form the alignment-methods lesson on identical data.

Deliberately simplified: pairs come from the current policy scored by the
same heuristic (self-labelled preferences, not human ones), sequence
log-probs are summed over all generated tokens, and ties are skipped.
"""
from __future__ import annotations

from mini_networks.models.rlhf.model import shakespearean_score  # noqa: F401
from mini_networks.models.transformer.model import TransformerLM  # noqa: F401
