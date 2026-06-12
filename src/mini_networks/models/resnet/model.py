"""Mini ResNet for 28x28 grayscale images.

ResNet's key idea (He et al., 2015): instead of asking a block to learn a
full mapping H(x), let it learn only the residual F(x) and add the input
back: out = F(x) + x. The identity shortcut gives gradients a direct path
around every block, so very deep networks stop degrading as depth grows.

Each BasicBlock here is the classic two-conv residual unit:

    x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+ shortcut) -> ReLU
         shortcut = Identity, or 1x1 Conv + BN when stride/channels change

Overall layout with base_channels=32:

    [B,1,28,28] -> stem Conv3x3+BN+ReLU (32ch)
                -> BasicBlock(32->32,  s=1)   28x28
                -> BasicBlock(32->64,  s=2)   14x14
                -> BasicBlock(64->128, s=2)    7x7
                -> global avg pool -> Linear(128->10)

Deliberately simplified vs the paper: 3 blocks instead of 4 stages of many
(ResNet-18 has 8 blocks, ResNet-50 has 16 bottlenecks), tiny channel widths,
no bottleneck (1x1-3x3-1x1) blocks, no 7x7 stem or max-pool (28x28 input is
already small), and no weight-decay/lr-schedule tricks from the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(self.conv1(x))
        out = torch.relu(out)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class MiniResNet(nn.Module):
    def __init__(self, base_channels: int = 32, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )
        self.layer1 = BasicBlock(base_channels, base_channels, stride=1)
        self.layer2 = BasicBlock(base_channels, base_channels * 2, stride=2)
        self.layer3 = BasicBlock(base_channels * 2, base_channels * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
