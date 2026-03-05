"""RNN language models: vanilla RNN, LSTM, and GRU.

All three share the same interface so they are drop-in interchangeable:

  logits, aux = model(tokens)           # train / eval forward pass
  output = model.generate(prompt, ...)  # autoregressive generation

Architecture
------------
  token_embed  [vocab_size → hidden_dim]
  cell         nn.RNN | nn.LSTM | nn.GRU  (n_layers, hidden_dim)
  dropout      applied between layers inside the cell (PyTorch built-in)
  lm_head      [hidden_dim → vocab_size]

The key educational difference from Transformer / Mamba:
  - Transformer processes all positions in parallel via attention
  - Mamba processes all positions in parallel via a parallel scan
  - RNN/LSTM/GRU process one token at a time, maintaining a hidden state
    → sequential by nature, but can maintain explicit memory across steps

generate() carries the hidden state between steps for efficient inference,
rather than reprocessing the full context each step.
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
