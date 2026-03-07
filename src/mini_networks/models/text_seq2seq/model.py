"""Minimal Transformer seq2seq."""
from __future__ import annotations

import torch
import torch.nn as nn


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, d_ff: int, seq_len: int):
        super().__init__()
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=d_ff,
            batch_first=True,
        )
        self.head = nn.Linear(d_model, vocab_size)
        self.seq_len = seq_len

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        B, S = src.shape
        _, T = tgt.shape
        src_pos = torch.arange(S, device=src.device).unsqueeze(0)
        tgt_pos = torch.arange(T, device=tgt.device).unsqueeze(0)
        src_e = self.src_embed(src) + self.pos(src_pos)
        tgt_e = self.tgt_embed(tgt) + self.pos(tgt_pos)
        out = self.transformer(src_e, tgt_e)
        return self.head(out)
