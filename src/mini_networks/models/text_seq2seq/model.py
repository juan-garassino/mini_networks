"""Minimal encoder-decoder Transformer for sequence-to-sequence text tasks.

Key idea: unlike a decoder-only LM, seq2seq separates reading from writing. An
encoder ingests the full source bidirectionally; a decoder generates the target
while attending to the encoder's output via cross-attention:
    cross_attn = softmax(Q_dec K_enc^T / sqrt(d_k)) V_enc,
so every target position can consult any source position — the original
"Attention Is All You Need" architecture.

This implementation: separate source and target embedding tables (vocab →
d_model) plus one shared learned positional embedding (seq_len → d_model), all
wrapped around torch's nn.Transformer with n_layers encoder layers, n_layers
decoder layers, n_heads heads, and d_ff feed-forward width (batch_first). The
head is Linear(d_model → vocab). The trainer splits each text sequence into
(src, tgt) halves and trains with shifted teacher forcing: the decoder reads
tgt[:, :-1] under a causal mask (generate_square_subsequent_mask, applied in
forward()) and is scored against tgt[:, 1:], so position i predicts token i+1
from tokens ≤ i only — no future leakage.

Deliberately simplified: no BOS/EOS tokens (the first target token is never
predicted; decoding seeds from the last source token), no padding masks, no
weight tying between embeddings and head, and greedy argmax decoding only.
"""
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
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=tgt.device)
        out = self.transformer(src_e, tgt_e, tgt_mask=causal)
        return self.head(out)
