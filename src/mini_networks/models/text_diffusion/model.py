"""Masked-diffusion language model — the LLaDA recipe at char scale.

Key idea (LLaDA, arXiv 2502.09992): a language model does not have to be
autoregressive. Define a forward process that MASKS each token independently
with probability t (t ~ U(0,1]), and train a plain bidirectional transformer
to predict the masked tokens. That objective is a variational bound on
likelihood; generation runs the process backwards — start from all-MASK and
iteratively unmask the highest-confidence positions over K rounds, refining
the whole sequence in parallel instead of committing left-to-right. At 8B
scale this matches strong AR baselines and beats them on "reversal" tasks
that left-to-right factorization struggles with.

This implementation (defaults): the same nano-transformer shape as the zoo's
`transformer` (d_model=128, 4 layers) with the causal mask REMOVED — that
one change is the whole story: "same architecture, two generation orders".
Vocab gains one MASK token. No time embedding: the mask ratio is implicit in
the input (LLaDA does the same).

Key equations:
  Training   t ~ U(t_min, 1); mask each position w.p. t;
             loss = (1/t) * mean CE over MASKED positions only  (the 1/t
             weight makes the masked-CE an upper bound on -log p(x))
  Sampling   K rounds (tier-capped like image-diffusion timesteps): predict
             all masked positions, unmask the top-confidence fraction so the
             sequence resolves gradually — low-confidence spots get revised
             in later rounds with more context filled in.

Deliberately simplified vs the paper: uniform-random masking only (no SFT /
semi-AR remasking schedules), confidence-based unmasking order, temperature
multinomial fill, and prompt tokens are pinned for prefix-conditioned
generation (the showcase contract: prompt in, text out).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextDiffusionLM(nn.Module):
    """Bidirectional transformer over char tokens + one MASK token."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_id = vocab_size  # appended MASK token
        self.token_embed = nn.Embedding(vocab_size + 1, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)  # never predicts MASK
        self.seq_len = seq_len

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, T] (may contain mask_id) -> logits [B, T, V]. No causal mask."""
        T = tokens.shape[1]
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.token_embed(tokens) + self.pos_embed(pos)
        x = self.blocks(x)
        return self.lm_head(self.norm(x))

    def masked_loss(
        self, tokens: torch.Tensor, t_min: float = 0.05
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One training step of the LLaDA bound: returns (loss, mask_used)."""
        B, T = tokens.shape
        t = torch.rand(B, 1, device=tokens.device) * (1.0 - t_min) + t_min
        mask = torch.rand(B, T, device=tokens.device) < t
        # guarantee at least one masked position per sequence
        mask[torch.arange(B), torch.randint(0, T, (B,), device=tokens.device)] = True
        corrupted = torch.where(mask, torch.full_like(tokens, self.mask_id), tokens)
        logits = self(corrupted)
        ce = F.cross_entropy(
            logits.view(-1, self.vocab_size), tokens.view(-1), reduction="none"
        ).view(B, T)
        per_seq = (ce * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        loss = (per_seq / t.squeeze(1)).mean()  # 1/t weighting (likelihood bound)
        return loss, mask

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        steps: int = 25,
    ) -> torch.Tensor:
        """Prefix-conditioned iterative unmasking. Returns [B, prompt+new]."""
        self.eval()
        B, P = prompt.shape
        total = min(P + max_new_tokens, self.seq_len)
        x = torch.full((B, total), self.mask_id, dtype=torch.long, device=prompt.device)
        x[:, :P] = prompt[:, :total]
        pinned = torch.zeros(B, total, dtype=torch.bool, device=prompt.device)
        pinned[:, :P] = True

        for step in range(steps):
            masked = x == self.mask_id
            if not masked.any():
                break
            logits = self(x) / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, self.vocab_size), 1
            ).view(B, total)
            conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            conf = conf.masked_fill(~masked | pinned, -1.0)
            # unmask an equal share of the remaining masks each round
            remaining_rounds = steps - step
            for b in range(B):
                n_masked = int(masked[b].sum().item())
                k = max(1, n_masked // remaining_rounds)
                top = torch.topk(conf[b], min(k, n_masked)).indices
                x[b, top] = sampled[b, top]
        x[x == self.mask_id] = 0  # safety: resolve any stragglers
        return x
