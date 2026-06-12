"""Transformer encoder for per-token classification (BERT-style tagging, in miniature).

Key idea: not every text model predicts the next token. Token classification
emits one label per input position — the pattern behind NER, POS tagging, and
span extraction. The crucial architectural difference from a language model is
the absence of a causal mask: the encoder is fully bidirectional, so the label
for position t is computed from the whole sequence, left and right context alike.

This implementation: token embedding (vocab → d_model) + learned positional
embedding (seq_len → d_model), summed and fed through an n_layers
nn.TransformerEncoder (n_heads heads, d_ff = 4 * d_model, dropout 0.1,
batch_first). The head is Linear(d_model → 2) applied at every position; the toy
task is tagging each character as vowel vs other, so the model can be sanity-
checked by eye. Output shape is [B, T, 2]; the trainer applies cross-entropy per
position: y_t = softmax(W h_t).

Key contrast to hold onto: TransformerLM masks attention and predicts a vocab-
sized distribution at each step; this model attends everywhere and predicts a
tiny label set at each step — same backbone, opposite information flow.

Deliberately simplified vs BERT-style taggers: characters instead of subwords, a
hard-coded 2-class head, no padding/attention masks, no [CLS]/[SEP] structure,
no pretraining — the encoder trains from scratch on the tagging objective, and
no CRF layer to model label-to-label transitions.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TokenClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, seq_len: int):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 2)  # vowel vs other

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token(x) + self.pos(pos)
        h = self.encoder(h)
        return self.head(h)
