"""Text preprocessing utilities."""
from __future__ import annotations

import torch


def split_seq_halves(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split sequence into two halves for seq2seq toy tasks."""
    mid = x.size(0) // 2
    return x[:mid], x[mid:]


def vowel_labels(tokens: torch.Tensor, itos: dict[int, str]) -> torch.Tensor:
    """Binary labels: vowel vs other."""
    vowels = set("aeiouAEIOU")
    labels = torch.zeros_like(tokens)
    for i, t in enumerate(tokens.tolist()):
        ch = itos.get(int(t), "")
        labels[i] = 1 if ch in vowels else 0
    return labels
