"""RLHF reward components: a heuristic scorer and an optional learned RewardModel.

Key idea: RLHF turns "which output is better" signals into a scalar reward, then
fine-tunes a pretrained LM with PPO to maximise it — while a KL penalty to a
frozen copy of the pretrained LM (the reference policy) stops the policy from
drifting into degenerate, reward-hacked text.

This implementation is reward-model-free in practice: the trainer scores rollouts
with shakespearean_score(text) — the fraction of words in a fixed archaic set
(thou, thee, hath, ...), so no human preference data is needed. The learned
alternative, RewardModel, wraps a frozen TransformerLM, reads its last hidden
state [:, -1, :] of size d_model, and maps it through Linear(d_model, 32) → Tanh
→ Linear(32, 1) to a scalar reward [B].

Key equations: Bradley-Terry preference loss = -log sigmoid(r_chosen - r_rejected),
which maximises P(chosen > rejected) under a logistic model; the PPO stage in
trainer.py optimises E[min(rho_t A, clip(rho_t, 1-eps, 1+eps) A)] - beta *
KL(pi || pi_ref), with rho_t the per-token prob ratio vs the frozen reference.

Deliberately simplified vs InstructGPT (Ouyang et al. 2022): the reward is a word-
counting heuristic, not a model trained on human rankings; the episode's single
scalar reward is spread uniformly over tokens (no value network or per-token
credit assignment); and the reward head pools only the final position rather than
all tokens.
"""
from __future__ import annotations

import copy
import re
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_networks.models.transformer.model import TransformerLM


# ---------------------------------------------------------------------------
# Heuristic reward
# ---------------------------------------------------------------------------

_ARCHAIC = {
    "thou", "thee", "thy", "thine", "dost", "doth", "hath",
    "hast", "shalt", "wilt", "art", "wherefore", "forsooth",
    "prithee", "mayst", "wouldst", "couldst", "shouldst",
    "methinks", "perchance", "ere", "tis", "twas", "nay",
    "alas", "hark", "lo", "fie", "verily",
}


_COMMON = {
    "the", "and", "to", "of", "i", "a", "in", "that", "is", "my", "you",
    "not", "with", "his", "he", "be", "your", "for", "have", "it", "we",
    "what", "me", "this", "so", "but", "him", "her", "our", "shall", "will",
    "good", "lord", "king", "love", "sir", "come", "let", "do", "no", "o",
}


def shakespearean_score(text: str) -> float:
    """DENSE heuristic quality score for Shakespearean text in [0, 1].

    The original score counted only exact archaic words — a mini char-LM's
    near-gibberish emits those with probability ~0, so every PPO rollout
    scored 0.0 and the policy had no gradient at all (audit reward:0.0; the
    same sparse-reward failure the maze had before shaping). Three graded
    bands keep the signal dense at every capability level:
      - real_frac:   emits actual English words (learnable from gibberish)
      - suffix_frac: archaic morphology (-eth/-est/-'st/'d)
      - archaic:     exact archaic-word hits (the true target, amplified)
    """
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0
    real_frac = sum(1 for w in words if w in _COMMON or w in _ARCHAIC) / len(words)
    suffix_frac = sum(1 for w in words if w.endswith(("eth", "est", "'st", "'d"))) / len(words)
    archaic = sum(1 for w in words if w in _ARCHAIC) / len(words)
    return min(1.0, 0.25 * real_frac + 0.25 * min(1.0, 4 * suffix_frac) + 0.5 * min(1.0, 5 * archaic))


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """Scalar reward model built on top of a frozen TransformerLM.

    The base LM is frozen — only the reward head is trained.
    """

    def __init__(
        self,
        base_lm: TransformerLM,
        hidden: int = 32,
    ):
        super().__init__()
        d_model = base_lm.token_embed.embedding_dim
        # Freeze base LM
        self.base_lm = base_lm
        for p in self.base_lm.parameters():
            p.requires_grad_(False)

        self.reward_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, T] → reward [B]"""
        # Extract last hidden state from base LM
        import torch
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.base_lm.token_embed(tokens) + self.base_lm.pos_embed(positions)
        for block in self.base_lm.blocks:
            x, _ = block(x)
        x = self.base_lm.norm(x)         # [B, T, d_model]
        last = x[:, -1, :]               # [B, d_model]
        return self.reward_head(last).squeeze(-1)  # [B]

    def bradley_terry_loss(
        self,
        chosen_tokens: torch.Tensor,
        rejected_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Ranking loss: -log(sigmoid(r_chosen - r_rejected))."""
        r_chosen   = self(chosen_tokens)
        r_rejected = self(rejected_tokens)
        return -F.logsigmoid(r_chosen - r_rejected).mean()
