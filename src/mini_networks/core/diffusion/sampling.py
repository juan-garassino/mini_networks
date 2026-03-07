"""Shared diffusion sampling loop with optional guidance hooks."""
from __future__ import annotations

from typing import Callable, Any

import torch


PredictFn = Callable[[torch.Tensor, torch.Tensor, int, dict[str, Any] | None], torch.Tensor]
GuidanceFn = Callable[
    [torch.Tensor, torch.Tensor, int, torch.Tensor, dict[str, Any] | None], torch.Tensor
]
StepCallback = Callable[[torch.Tensor, int, int, dict[str, Any] | None], torch.Tensor | None]


def sample_loop(
    *,
    scheduler,
    predict_noise: PredictFn,
    shape: tuple[int, ...],
    device: str | torch.device,
    timesteps: int,
    seed: int | None = None,
    guidance_fn: GuidanceFn | None = None,
    step_callback: StepCallback | None = None,
    state: dict[str, Any] | None = None,
    logger: Any | None = None,
    log_every: int = 0,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(shape, device=device)
    batch = shape[0]
    for step_idx, t in enumerate(reversed(range(timesteps))):
        t_batch = torch.full((batch,), t, device=device, dtype=torch.long)
        eps = predict_noise(x, t_batch, t, state)
        if guidance_fn is not None:
            eps = guidance_fn(x, t_batch, t, eps, state)
        x = scheduler.step(eps, t, x)
        if logger is not None and log_every and step_idx % log_every == 0:
            logger.log_metrics(step_idx, {"sample_t": int(t), "step": step_idx})
        if step_callback is not None:
            updated = step_callback(x, t, step_idx, state)
            if updated is not None:
                x = updated
    return x
