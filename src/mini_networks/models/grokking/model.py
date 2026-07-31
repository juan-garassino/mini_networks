"""Grokking: generalization beyond overfitting on modular arithmetic.

Key idea (Power et al., arXiv 2201.02177): train a small transformer on a
symbolic task — division mod a prime, shown as `a op b =` token sequences with
half of all pairs held out — and watch two curves: train accuracy hits 100%
within ~10^3 steps, while VALIDATION accuracy sits at chance for 10x-1000x
longer and then abruptly jumps to ~100%. The network first memorizes, then —
under continued optimization pressure with strong weight decay — reorganizes
into the general algorithm. Generalization is a phase transition here, not a
gradual accompaniment of training.

This implementation (defaults): a 2-layer causal transformer (d_model=128,
4 heads) reads `[a, op, b, =]` and classifies the answer among p=97 residues
from the final position. The training knobs that matter are in the trainer:
AdamW with weight_decay=1.0 (grokking's key ingredient — at wd~0 the jump can
take orders of magnitude longer or never come), near-full-batch training, and
a step budget (not epochs) since the dataset is only ~4.7k train pairs.

Key equation: the task is a / b ≡ a · b^(p-2) (mod p) by Fermat's little
theorem — nothing about the mapping is learnable "locally"; the model must
discover modular structure to generalize to held-out pairs.

Deliberately simplified vs the paper: single operation (division, the
headline task) instead of the full operation zoo, and no optimizer ablations
— the zoo's story is the delayed-generalization curve itself.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GrokkingTransformer(nn.Module):
    """Tiny causal transformer; answer read from the last position."""

    SEQ_LEN = 4  # [a, op, b, =]

    def __init__(
        self,
        p: int = 97,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.p = p
        vocab = p + 2  # residues + op + "="
        self.token_embed = nn.Embedding(vocab, d_model)
        self.pos_embed = nn.Embedding(self.SEQ_LEN, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, 4] -> logits [B, p] over the answer residue."""
        T = tokens.shape[1]
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.token_embed(tokens) + self.pos_embed(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=tokens.device)
        x = self.blocks(x, mask=mask, is_causal=True)
        return self.head(self.norm(x[:, -1]))
