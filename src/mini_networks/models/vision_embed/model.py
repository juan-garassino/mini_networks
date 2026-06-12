"""Simple CNN encoder for embeddings.

An embedding model maps an image to a point on the unit hypersphere so
that similarity becomes geometry: with L2-normalised vectors, cosine
similarity is just the dot product z1 . z2, and "similar images" means
"nearby embeddings". This is the building block behind retrieval, metric
learning, and the image tower of CLIP-style models — the network's job is
not to classify but to place inputs in a useful coordinate system.

This implementation (embed_dim=64):

    [B,1,28,28] -> ConvBNReLU(1->32) -> MaxPool2 -> [B,32,14,14]
                -> ConvBNReLU(32->64) -> MaxPool2 -> [B,64,7,7]
                -> Flatten(3136) -> Linear(3136->64)
                -> L2 normalise -> [B,64] unit vectors

The final F.normalize is what makes the output an embedding rather than a
feature vector: every output has norm 1, so training objectives and
nearest-neighbour search can rely on cosine similarity alone.

Deliberately simplified: the same two-conv backbone as the SmallCNN
classifier with a bare linear projection — no projection MLP, no deep
backbone, no temperature scaling. The model defines only the mapping; the
training signal (contrastive pairs, classification proxy, CLIP pairing)
is supplied by whichever trainer or composition uses it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_networks.core.blocks.cnn import ConvBNReLU


class VisionEmbedCNN(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(1, 32),
            nn.MaxPool2d(2),
            ConvBNReLU(32, 64),
            nn.MaxPool2d(2),
        )
        self.proj = nn.Linear(64 * 7 * 7, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        z = self.proj(h)
        return F.normalize(z, dim=-1)
