"""Config for the grokking study model."""
from __future__ import annotations

from mini_networks.core.config import BaseConfig


class GrokkingConfig(BaseConfig):
    """Grokking (arXiv 2201.02177) as a first-class zoo entry.

    A tiny 2-layer transformer on modular division: train accuracy saturates
    within ~1k steps, validation accuracy stays at chance for orders of
    magnitude longer, then suddenly jumps to ~1.0 — generalization long after
    overfitting. The metrics.jsonl train/val curve IS the deliverable; the
    playground renders it natively.
    """

    model_name: str = "grokking"
    dataset: str = "modular_arithmetic"

    p: int = 97                 # prime modulus; task is a / b (mod p), vocab p+2
    train_frac: float = 0.5     # fraction of all pairs in train (paper's key knob)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2           # paper scale — smallest model in the zoo
    dropout: float = 0.0

    n_train_steps: int = 100_000  # step-based, not epoch-based (limit_steps caps per tier)
    weight_decay: float = 1.0     # grokking's key ingredient (AdamW wd ~ 1)
    batch_size: int = 512         # near-full-batch, as in the paper
    log_every: int = 50           # metrics cadence — the curve needs dense points
