"""Recurrent language models: one wrapper class over nn.RNN, nn.LSTM, and nn.GRU.

Key idea: instead of attending over all previous tokens, a recurrent network
compresses the entire history into a fixed-size hidden state that is updated one
token at a time. The cell_type config flag ("rnn" | "lstm" | "gru") selects the
recurrence, everything else is identical.

This implementation (defaults): token embedding vocab → hidden_dim=256, then a
2-layer recurrent cell (batch_first, inter-layer dropout), then lm_head
Linear(256 → vocab). forward(tokens [B, T]) returns (logits [B, T, V],
aux_loss=0.0) — the same API as TransformerLM and NanoMamba, so trainers are
fully interchangeable.

Key equations: vanilla RNN h_t = tanh(W_x x_t + W_h h_{t-1} + b); LSTM adds gates
f, i, o and a cell state c_t = f_t * c_{t-1} + i_t * tanh(g_t) so gradients can
flow through c unchanged; GRU merges this into update/reset gates with a single
state.

The educational payoff is in generate(): it warms the hidden state up on the
prompt once, then feeds back one token at a time, carrying (h, c) between steps —
each new token costs O(1) instead of the Transformer's full-context recompute.
Deliberately simplified: cuDNN-fused PyTorch cells rather than hand-written gate
math, zero-initialised hidden state, no truncated-BPTT state carrying between
training batches, and sampling is temperature-only (no top-k / nucleus).
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


_CELL_MAP = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


class RNNLanguageModel(nn.Module):
    """Language model backed by a vanilla RNN, LSTM, or GRU cell.

    forward() returns (logits [B, T, V], aux_loss=0.0) — same API as
    TransformerLM and NanoMamba so trainers are fully interchangeable.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        seq_len: int = 128,
        dropout: float = 0.1,
        cell_type: Literal["rnn", "lstm", "gru"] = "lstm",
    ):
        super().__init__()
        if cell_type not in _CELL_MAP:
            raise ValueError(f"cell_type must be one of {list(_CELL_MAP)}; got {cell_type!r}")

        self.cell_type = cell_type
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.seq_len = seq_len

        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.cell = _CELL_MAP[cell_type](
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def _init_hidden(self, batch_size: int, device: torch.device):
        """Zero initial hidden state (and cell state for LSTM)."""
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        if self.cell_type == "lstm":
            return (h, torch.zeros_like(h))
        return h

    def forward(
        self,
        tokens: torch.Tensor,
        hidden=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: [B, T] long
            hidden: optional initial hidden state (reuse across chunks for generation)
        Returns:
            logits [B, T, V], aux_loss=0.0
        """
        B = tokens.size(0)
        if hidden is None:
            hidden = self._init_hidden(B, tokens.device)

        x = self.token_embed(tokens)          # [B, T, hidden_dim]
        out, _ = self.cell(x, hidden)         # [B, T, hidden_dim]
        out = self.drop(out)
        return self.lm_head(out), torch.tensor(0.0, device=tokens.device)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive generation carrying hidden state between steps.

        Unlike Transformer/Mamba (which re-process the full context), the RNN
        maintains a running hidden state — each new token only requires one step.
        """
        self.eval()
        B = prompt.size(0)

        # Warm up hidden state on the prompt
        x = self.token_embed(prompt)
        hidden = self._init_hidden(B, prompt.device)
        _, hidden = self.cell(x, hidden)

        # Generate token by token, feeding back the previous output
        generated = prompt
        last_token = prompt[:, -1:]
        for _ in range(max_new_tokens):
            x = self.token_embed(last_token)      # [B, 1, H]
            out, hidden = self.cell(x, hidden)    # [B, 1, H]
            logits = self.lm_head(self.drop(out[:, -1, :]))  # [B, V]
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            generated = torch.cat([generated, next_token], dim=1)
            last_token = next_token

        return generated
