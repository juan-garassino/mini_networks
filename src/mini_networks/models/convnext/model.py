"""Tiny ConvNeXt-like model for 28x28 grayscale images.

ConvNeXt (Liu et al., 2022) asks: how much of a Vision Transformer's edge
comes from attention, and how much from its macro-design? It "modernises"
a ResNet with Transformer-era choices — depthwise conv as a stand-in for
token mixing, an inverted bottleneck MLP (expand 4x, like a Transformer
FFN), LayerNorm instead of BatchNorm, GELU instead of ReLU — and matches
ViTs without any attention.

Each ConvNeXtBlock here follows that recipe:

    x -> depthwise Conv3x3 -> LayerNorm (channels-last permute)
      -> Conv1x1 (dim -> 4*dim) -> GELU -> Conv1x1 (4*dim -> dim) -> + x

Overall layout with base_channels=32:

    [B,1,28,28] -> ConvBNReLU stem (32ch)        28x28
                -> ConvNeXtBlock(32)             28x28
                -> Conv2x2 s=2 downsample (64)   14x14
                -> ConvNeXtBlock(64)
                -> Conv2x2 s=2 downsample (128)   7x7
                -> ConvNeXtBlock(128)
                -> global avg pool -> Linear(128->10)

Deliberately simplified: one block per stage (real ConvNeXt-T uses
(3,3,9,3) blocks at 96-768 channels), 3x3 depthwise kernels instead of the
paper's 7x7, a BN+ReLU stem rather than the patchify stem, and no
layer-scale, stochastic depth, or EMA training tricks.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mini_networks.core.blocks.cnn import ConvBNReLU


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw1 = nn.Conv2d(dim, dim * 4, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * 4, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dw(x)
        # LayerNorm expects channels-last
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)
        out = self.pw2(self.act(self.pw1(out)))
        return x + out


class TinyConvNeXt(nn.Module):
    def __init__(self, base_channels: int = 32, num_classes: int = 10):
        super().__init__()
        c = base_channels
        self.stem = ConvBNReLU(1, c, k=3, s=1, p=1)
        self.stage1 = ConvNeXtBlock(c)
        self.down1 = nn.Conv2d(c, c * 2, 2, stride=2)
        self.stage2 = ConvNeXtBlock(c * 2)
        self.down2 = nn.Conv2d(c * 2, c * 4, 2, stride=2)
        self.stage3 = ConvNeXtBlock(c * 4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
