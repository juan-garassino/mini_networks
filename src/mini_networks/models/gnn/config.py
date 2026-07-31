"""Config for the graph convolutional network."""
from __future__ import annotations

from mini_networks.core.config import BaseConfig


class GNNConfig(BaseConfig):
    """Mini GCN (Kipf & Welling 2016) — the zoo's first graph-modality model.

    Transductive node classification on a seeded stochastic block model: the
    labels are carried by the ADJACENCY (community structure), while node
    features are a deliberately weak signal. The trainer logs a feature-only
    MLP baseline every eval, so the gate provably measures message passing,
    not feature leakage.
    """

    model_name: str = "gnn"
    dataset: str = "synthetic_graph"

    n_nodes: int = 200
    n_communities: int = 4
    p_in: float = 0.15         # intra-community edge probability (detectability knob)
    p_out: float = 0.02        # inter-community edge probability
    n_features: int = 8        # first n_communities dims carry the weak signal
    feat_signal: float = 0.3   # weak on purpose: feature-only MLP must stay far below the bar
    train_per_class: int = 5   # stratified train mask (20 labeled nodes total)

    hidden_dim: int = 32
    dropout: float = 0.5
    epochs: int = 400          # one epoch == ONE full-batch step (len-1 dataset)
