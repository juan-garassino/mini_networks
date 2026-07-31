"""Config for the mini Segment Anything model."""
from __future__ import annotations

from mini_networks.core.config import BaseConfig


class SAMConfig(BaseConfig):
    """Mini SAM (arXiv 2304.02643) as a zoo entry.

    Promptable segmentation on TWO-digit composites: the image alone is
    ambiguous — the click/box selects which digit to segment. evaluate()
    reports both the clicked-digit IoU (the gate metric) and the IoU against
    the OTHER digit's mask; the gap is the evidence that the prompt is
    actually used.
    """

    model_name: str = "sam"
    dataset: str = "sam_two_digit"

    embed_dim: int = 64        # image/prompt token width
    n_heads: int = 4
    n_decoder_layers: int = 2  # two-way attention rounds (paper: 2)
    n_masks: int = 3           # ambiguity-aware multi-mask output (paper: 3)
    neg_prompt_prob: float = 0.3  # chance of adding a negative click (on the other digit)
    box_prompt_prob: float = 0.3  # chance of using a box prompt instead of a point
