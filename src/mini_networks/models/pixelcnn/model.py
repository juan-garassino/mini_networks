"""Minimal PixelCNN with masked convolutions (grayscale)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ("A", "B")
        self.register_buffer("mask", torch.ones_like(self.weight))
        _, _, h, w = self.weight.shape
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :, :] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class PixelCNN(nn.Module):
    def __init__(self, n_filters: int = 32, n_layers: int = 4):
        super().__init__()
        layers = [
            MaskedConv2d("A", 1, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        for _ in range(n_layers - 1):
            layers += [
                MaskedConv2d("B", n_filters, n_filters, kernel_size=3, padding=1),
                nn.ReLU(),
            ]
        layers.append(nn.Conv2d(n_filters, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
