"""Tiny MobileNet-like model (depthwise separable conv)."""
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
