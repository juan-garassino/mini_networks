"""Small CNN classifier for 28x28 grayscale images.

The plain convolutional network: stacked conv layers learn local filters
(edges, strokes, curves) whose receptive field grows with depth, and pooling
makes the features progressively translation-invariant. The key idea is
weight sharing — one 3x3 kernel is slid over the whole image, so the layer
has ~few-hundred parameters instead of the millions a dense layer would need.

Architecture of this implementation:

    [B,1,28,28] -> ConvBNReLU(1->32) -> MaxPool2 -> [B,32,14,14]
                -> ConvBNReLU(32->64) -> MaxPool2 -> [B,64,7,7]
                -> Flatten(3136) -> Linear(3136->64) -> ReLU
                -> Linear(64->10 logits)

Each ConvBNReLU is Conv2d(3x3) -> BatchNorm -> ReLU; BatchNorm normalizes
activations per channel, which stabilizes and speeds up training.

Deliberately simplified: only two conv stages and one small hidden layer
(LeNet-scale, fine for MNIST), no dropout or data augmentation, no global
average pooling — the flatten ties the head to a fixed 28x28 input size.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mini_networks.core.blocks.cnn import ConvBNReLU


class SmallCNN(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(1, 32),
            nn.MaxPool2d(2),
            ConvBNReLU(32, 64),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
