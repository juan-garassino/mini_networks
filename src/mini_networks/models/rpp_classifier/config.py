"""Config for the Residual Pathway Prior classifier."""
from __future__ import annotations

from mini_networks.core.config import BaseConfig


class RPPClassifierConfig(BaseConfig):
    """Residual Pathway Priors (arXiv 2112.01388) as a zoo entry.

    Each block sums a CONSTRAINED pathway (conv — translation-equivariant) and
    a FREE pathway (full linear over the flattened feature map — contains conv
    as a special case). Different L2 prior strengths make the model PREFER the
    structured solution while staying able to absorb symmetry violations:
    inductive bias as a soft prior, not a hard architectural constraint.
    """

    model_name: str = "rpp_classifier"
    dataset: str = "mnist"
    num_classes: int = 10

    hidden_dim: int = 16       # conv channels of the second block
    prior_conv: float = 1e-4   # weak L2 on the conv path  (wide prior — cheap to use)
    prior_mlp: float = 1e-2    # strong L2 on the free path (narrow prior — pay to deviate)
