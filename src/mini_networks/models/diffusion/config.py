from __future__ import annotations
from typing import Literal
from mini_networks.core.config import BaseConfig


class DiffusionConfig(BaseConfig):
    model_name: str = "diffusion"
    timesteps: int = 1000
    schedule: Literal["linear", "cosine"] = "linear"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    image_size: int = 28
    in_channels: int = 1
    base_channels: int = 32
    dataset: str = "mnist"

    # EMA (Exponential Moving Average) of model weights — improves sample quality.
    # 0.995, NOT the paper's 0.9999: with M-tier's ~1000 steps, 0.9999 leaves
    # the EMA ≈90% initial random weights (0.9999^1000≈0.905) and
    # load_checkpoint PREFERS model_ema.pt — the gate was scoring a
    # near-random model (noise samples, 2026-07-11 audit). 0.995 converges in
    # a few hundred steps.
    ema_decay: float = 0.995    # set to 0.0 to disable EMA

    # Curriculum learning — train on harder (higher-variance) images first
    curriculum: bool = False    # if True, sort batches by image complexity

    # LR warmup — ramp learning rate from 0 to lr over warmup_steps
    warmup_steps: int = 0       # set > 0 to enable warmup
