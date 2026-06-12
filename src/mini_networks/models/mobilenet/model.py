"""Tiny MobileNet-like model (depthwise separable conv).

MobileNet's key idea (Howard et al., 2017): factor a standard convolution
into a depthwise conv (one 3x3 filter per channel, no cross-channel mixing)
followed by a pointwise 1x1 conv (mixes channels, no spatial extent). Cost
drops from k*k*Cin*Cout to k*k*Cin + Cin*Cout per position — roughly an
8-9x saving for 3x3 kernels — with little accuracy loss.

This implementation with width_mult=1.0:

    [B,1,28,28] -> stem Conv3x3+BN+ReLU (32ch)          28x28
                -> DepthwiseSeparable(32->64,  s=2)      14x14
                -> DepthwiseSeparable(64->128, s=2)       7x7
                -> global avg pool -> Linear(128->10)

Each DepthwiseSeparable block is depthwise Conv3x3 -> BN -> ReLU ->
pointwise Conv1x1 -> BN -> ReLU. The width_mult knob scales every channel
count, mirroring the paper's width multiplier alpha.

Deliberately simplified: 2 separable blocks instead of MobileNetV1's 13,
no resolution multiplier, and none of the V2/V3 additions (inverted
residuals, linear bottlenecks, squeeze-excite, hard-swish). MNIST-scale
channels throughout.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mini_networks.core.blocks.cnn import DepthwiseSeparable


class TinyMobileNet(nn.Module):
    def __init__(self, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()
        c = int(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(),
        )
        self.block1 = DepthwiseSeparable(c, c * 2, stride=2)
        self.block2 = DepthwiseSeparable(c * 2, c * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
