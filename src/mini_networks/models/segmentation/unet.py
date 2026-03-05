"""PyTorch UNet for binary and multiclass segmentation on MNIST."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegUNet(nn.Module):
    """
    Lightweight UNet for 28x28 inputs.
    out_channels=1 → binary (sigmoid), >1 → multiclass (softmax).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels

        self.enc1 = ConvBlock(in_channels, c)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(c, c * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c * 2, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)

        self.out_conv = nn.Conv2d(c, out_channels, 1)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = F.interpolate(d2, size=e2.shape[-2:])
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[-2:])
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.out_conv(d1)
        if self.out_channels == 1:
            return torch.sigmoid(logits)
        return logits  # softmax applied in loss


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Dice loss for binary segmentation."""
    pred_flat = pred.view(-1)
    target_flat = target.float().view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def multiclass_dice_loss(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int, smooth: float = 1.0
) -> torch.Tensor:
    """Averaged dice loss over all classes for multiclass segmentation."""
    pred_soft = F.softmax(pred, dim=1)
    losses = []
    for c in range(num_classes):
        p = pred_soft[:, c]
        t = (target == c).float()
        inter = (p * t).sum()
        losses.append(1 - (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth))
    return torch.stack(losses).mean()
