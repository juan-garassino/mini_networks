"""Residual Pathway Priors: soft equivariance instead of hard constraints.

Key idea (Finzi, Benton & Wilson, arXiv 2112.01388): don't CHOOSE between a
constrained architecture (convolution = translation equivariance) and a
flexible one (a full linear map over the image) — sum them, and encode the
preference in the PRIOR. Each RPP block computes conv(x) + free(x), where the
free path is a dense linear over the flattened feature map (it strictly
contains the conv as a special case). A weak L2 penalty on the conv path and
a strong one on the free path is a MAP estimate under Gaussian priors with
different variances: the model uses the equivariant solution where the
symmetry holds and pays a controlled price to break it where it doesn't.
When the symmetry is exact you match the constrained model; when it's
approximate or wrong, you win.

This implementation (defaults): two RPP blocks on downsampled MNIST
(14x14 then 7x7 — keeps the free path's dense matrix small), BatchNorm+ReLU
after the pathway sum, global pooling head. `prior_penalty()` returns the
per-pathway L2 terms (added to the loss by the trainer — the explicit-penalty
form IS the paper's prior, more faithful than optimizer weight decay);
`pathway_norms()` reports which pathway carried the solution — on clean MNIST
the conv path should dominate.

Deliberately simplified vs the paper: two blocks and translation symmetry
only (no EMLP/group machinery, no RL/dynamics tasks); symmetry breaking is
explored via the zoo's dataset flavors rather than bespoke datasets.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RPPBlock(nn.Module):
    """conv(x) + free(x) with per-pathway prior strengths recorded."""

    def __init__(self, in_ch: int, out_ch: int, spatial: int):
        super().__init__()
        self.out_ch = out_ch
        self.spatial = spatial
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.free = nn.Linear(in_ch * spatial * spatial, out_ch * spatial * spatial, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        c = self.conv(x)
        f = self.free(x.flatten(1)).view(B, self.out_ch, self.spatial, self.spatial)
        return F.relu(self.bn(c + f))


class RPPClassifier(nn.Module):
    def __init__(self, hidden_dim: int = 16, num_classes: int = 10):
        super().__init__()
        self.pool_in = nn.AvgPool2d(2)                     # 28 -> 14
        self.block1 = RPPBlock(1, hidden_dim // 2, spatial=14)
        self.pool_mid = nn.AvgPool2d(2)                    # 14 -> 7
        self.block2 = RPPBlock(hidden_dim // 2, hidden_dim, spatial=7)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool_in(x)
        x = self.block1(x)
        x = self.pool_mid(x)
        x = self.block2(x)
        x = x.mean(dim=(2, 3))  # global average pool
        return self.head(x)

    def prior_penalty(self, prior_conv: float, prior_mlp: float) -> torch.Tensor:
        """MAP penalty: (1/2sigma^2)||w||^2 per pathway, different sigmas."""
        conv_sq = sum(b.conv.weight.pow(2).sum() for b in (self.block1, self.block2))
        free_sq = sum(b.free.weight.pow(2).sum() for b in (self.block1, self.block2))
        return prior_conv * conv_sq + prior_mlp * free_sq

    @torch.no_grad()
    def pathway_norms(self) -> dict:
        """Which pathway carried the solution? (conv should win on clean MNIST)"""
        return {
            "conv_norm": float(sum(b.conv.weight.norm() for b in (self.block1, self.block2))),
            "free_norm": float(sum(b.free.weight.norm() for b in (self.block1, self.block2))),
        }
