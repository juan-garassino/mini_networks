"""Graph Convolutional Network: learning from structure, not just features.

Key idea (Kipf & Welling, arXiv 1609.02907): to classify a node, average
your neighbors. One GCN layer computes H' = act(Â H W) where
Â = D^-1/2 (A + I) D^-1/2 is the symmetrically normalized adjacency with
self-loops — each node's new representation is a degree-weighted mean of its
neighborhood's features pushed through a shared linear map. Stack two layers
and information flows two hops. On a community graph this is decisive: a
single node's features may be too noisy to classify, but the average over
~10 same-community neighbors is not — the graph itself denoises.

This implementation (defaults): 2 dense-matmul GCN layers (8 → 32 → 4) on a
200-node stochastic block model with only 5 labeled nodes per community
(transductive semi-supervised learning, the classic GCN setting). Everything
is dense torch — no PyG, no sparse ops; the message-passing equation IS the
code.

Key equation: H1 = ReLU(Â X W1), logits = Â H1 W2, loss = CE on the few
labeled nodes only; test nodes are classified by structure + propagation.

Deliberately simplified vs the literature: no attention (GAT), no sampling
(GraphSAGE), no edge features; dense N x N adjacency (fine at N=200, and the
honest reading of the math).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(
        self,
        n_features: int = 8,
        hidden_dim: int = 32,
        n_classes: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.w1 = nn.Linear(n_features, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_classes, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        """x [N, F], a_norm [N, N] -> logits [N, C]."""
        h = F.relu(a_norm @ self.w1(x))
        h = self.drop(h)
        return a_norm @ self.w2(h)
