"""Shared EMA (exponential moving average) of model weights."""
from __future__ import annotations

import copy

import torch


class EMA:
    """Exponential Moving Average of model parameters.

    Educational note (from legacy/013):
      EMA smooths weight updates across training steps.
      At inference we use the EMA model instead of the live model —
      EMA weights tend to produce sharper, less noisy samples.

    Usage:
      ema = EMA(model, decay=0.9999)
      # inside training loop:
      ema.update(model)
      # at checkpoint time:
      torch.save(ema.state_dict(), ...)

    Consumers: models/diffusion (denoiser EMA), models/gan (generator EMA).
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for shadow_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            shadow_p.copy_(self.decay * shadow_p + (1.0 - self.decay) * model_p)
        # Buffers too (BatchNorm running stats): parameters-only EMA left the
        # shadow's BN stats at their init values forever — the DCGAN
        # generator's EMA checkpoint normalized with garbage stats and
        # sampled uniform checkerboard (m-vision-8). Buffers track the live
        # model directly (no averaging — running stats are already an EMA).
        for shadow_b, model_b in zip(self.shadow.buffers(), model.buffers()):
            shadow_b.copy_(model_b)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()
