"""Four audio classifiers — one task, four input representations.

Key idea: the interesting choice in audio ML is usually the representation, not
the architecture. This file holds the same classification head behind four front
ends: raw waveform (1D conv), linear spectrogram (2D conv), mel spectrogram
(2D conv), and spectrogram frames as a token sequence (Transformer encoder).
Comparing them on one dataset shows what each representation buys.

This implementation: AudioCNN runs Conv1d over the raw wave [B, 1, T] with
channels 1 → 32 → 64 → 128, each kernel 5 / stride 2 (so T shrinks 8x), then
AdaptiveAvgPool1d(1) and Linear(128 → n_classes). AudioSpecCNN and AudioMelSpecCNN
are architecturally identical 2D CNNs (1 → 16 → 32 → 64, 3x3 convs with 2x2
max-pools, AdaptiveAvgPool2d, Linear(64 → n_classes)) — only the input features
differ, which is the point: a (mel-)spectrogram is just an image of time vs
frequency. AudioTransformer treats each spectrogram frame [B, T, D] as a token:
Linear(D → d_model=64), a 2-layer nn.TransformerEncoder (4 heads), mean-pool over
time, Linear(64 → n_classes).

Deliberately simplified vs production audio models: the STFT/mel extraction lives
in the data pipeline, not here; global average pooling discards temporal order in
the CNNs; no SpecAugment, no log-mel normalisation choices, no pretrained
self-supervised features (wav2vec 2.0 et al.), and the Transformer has no
positional encoding — mean pooling makes it order-invariant.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AudioCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).squeeze(-1)
        return self.head(h)


class AudioSpecCNN(nn.Module):
    """2D CNN for spectrograms."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).view(x.size(0), -1)
        return self.head(h)


class AudioMelSpecCNN(nn.Module):
    """2D CNN for mel-spectrograms."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).view(x.size(0), -1)
        return self.head(h)


class AudioTransformer(nn.Module):
    """Transformer encoder over spectrogram frames."""

    def __init__(self, input_dim: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, n_classes: int = 10):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        h = self.encoder(self.proj(x))
        pooled = h.mean(dim=1)
        return self.cls(pooled)
