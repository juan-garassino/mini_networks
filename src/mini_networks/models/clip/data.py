"""Image-text pair dataset for CLIP (re-exported from core data registry)."""
from __future__ import annotations

from mini_networks.core.data.registry import (
    DIGIT_CAPTIONS,
    MNISTImageTextDataset,
    label_to_all_tokens,
    label_to_tokens,
)

__all__ = [
    "DIGIT_CAPTIONS",
    "MNISTImageTextDataset",
    "label_to_tokens",
    "label_to_all_tokens",
]
