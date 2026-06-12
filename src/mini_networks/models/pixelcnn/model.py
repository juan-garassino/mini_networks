"""Minimal PixelCNN with masked convolutions (grayscale).

PixelCNN (van den Oord et al., 2016) is an autoregressive image model: the
joint distribution over pixels factorises into a product of conditionals
in raster-scan order,

    p(x) = prod_i p(x_i | x_1, ..., x_{i-1})

and the whole network is just a conv net whose kernels are masked so pixel
i can never see itself or any pixel after it. MaskedConv2d zeroes the
kernel weights at and right of centre on the middle row and all rows below.
Mask "A" (first layer) also hides the centre pixel itself — otherwise the
model could copy its input; mask "B" (later layers) allows the centre,
because by then that position holds features of preceding pixels only.

This implementation (n_filters=32, n_layers=4):

    [B,1,28,28] -> MaskedConv "A" 3x3 (1->32) -> ReLU
                -> 3 x [MaskedConv "B" 3x3 (32->32) -> ReLU]
                -> Conv1x1 (32->1)   per-pixel logit

Training is one parallel forward pass (every conditional is computed at
once thanks to the masks); sampling is inherently sequential — 784 forward
passes, one pixel at a time.

Deliberately simplified: a single logit per pixel (Bernoulli on binarised
MNIST) instead of a 256-way softmax, a small kernel and few layers (the
receptive field has the original's blind spot — the stacked 3x3 masks miss
part of the upper-right context, fixed in Gated PixelCNN by separate
vertical/horizontal stacks), and no residual connections or gating.
"""
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
