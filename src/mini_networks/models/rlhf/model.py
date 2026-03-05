"""RLHF components: RewardModel and heuristic Shakespearean scorer.

Architecture
-----------
  RewardModel wraps a pretrained TransformerLM and adds a scalar reward head:

    base_lm  (TransformerLM, frozen during RLHF)
      ↓ last hidden state [:, -1, :]       [B, d_model]
    reward_head  Linear(d_model, hidden) → Tanh → Linear(hidden, 1)
      ↓                                    [B, 1]

  The reward head is trained with a Bradley-Terry ranking loss on preference
  pairs (chosen vs rejected response), or you can substitute a heuristic.

Heuristic reward (no human labels needed)
-----------------------------------------
  `shakespearean_score(text, tokenizer)` counts archaic vocabulary
  (thou, thee, thy, dost, hath, etc.) as a proxy for "Shakespearean quality".
  Score is normalised by token count to be length-independent.

  This is the approach from legacy/012, adapted to our CharTokenizer.
  It lets the model learn without any human annotation.

Educational notes
-----------------
  - Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
    Maximises P(chosen > rejected) under the logistic model.
  - KL penalty: prevents the fine-tuned policy from drifting too far
    from the reference (pretrained) policy → avoids reward hacking.
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


def shakespearean_score(text: str) -> float:
    """Heuristic quality score for Shakespearean text.

    Counts archaic words as a fraction of total words.
    Returns a float in roughly [0, 1].
    """
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in _ARCHAIC)
    return hits / len(words)


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
