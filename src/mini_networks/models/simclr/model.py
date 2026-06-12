"""Simple SimCLR encoder + projection head.

SimCLR (Chen et al., 2020) learns visual representations without labels:
take two random augmentations of the same image (a "positive pair"), embed
both, and train so the pair agrees while all other images in the batch
("negatives") are pushed apart. The InfoNCE loss does this as a
classification problem over the batch:

    loss = -log( exp(sim(z_i, z_j)/T) / sum_k exp(sim(z_i, z_k)/T) )

where sim is cosine similarity (embeddings are L2-normalised, so the dot
product suffices) and T is a temperature. info_nce_loss builds the [2B,2B]
similarity matrix, masks the diagonal (self-similarity) with -1e9, and uses
cross-entropy with each view's partner as the target row.

This implementation:

    [B,1,28,28] -> Conv3x3(1->32)+ReLU -> MaxPool2
                -> Conv3x3(32->64)+ReLU -> MaxPool2
                -> Flatten(3136) -> Linear(3136->128) -> ReLU   (encoder h)
    h -> Linear(128->128) -> ReLU -> Linear(128->64) -> L2-normalise  (z)

The projection head is the paper's key empirical trick: contrast in z,
but keep h as the representation for downstream tasks — the head absorbs
augmentation-invariance distortion that would hurt h.

Deliberately simplified: a 2-conv encoder instead of ResNet-50, batch-size
negatives in the dozens rather than the paper's 4096, temperature 0.2, and
forward() returns the normalised z directly (h is not exposed separately).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLREncoder(nn.Module):
    def __init__(self, proj_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z = self.proj(h)
        return F.normalize(z, dim=-1)


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Compute SimCLR InfoNCE loss for a batch."""
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.matmul(z, z.T) / temperature  # [2B, 2B]
    sim.fill_diagonal_(-1e9)
    targets = torch.arange(batch_size, device=z.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)
    loss = F.cross_entropy(sim, targets)
    return loss
