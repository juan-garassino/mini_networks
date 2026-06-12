"""DDPM noise scheduler: the forward corruption process and the reverse denoising step.

Key idea: diffusion defines a fixed Markov chain that gradually destroys an image
with Gaussian noise over T steps, governed by variances beta_t. Because Gaussians
compose, any x_t is reachable from x_0 in one jump:
    q(x_t | x_0):  x_t = sqrt(a_bar_t) * x_0 + sqrt(1 - a_bar_t) * eps,
where a_t = 1 - beta_t and a_bar_t = prod_{s<=t} a_s. Generation inverts the chain
one step at a time using the model's noise prediction.

This implementation: timesteps=1000 by default with either a linear beta schedule
(1e-4 → 0.02) or the cosine schedule of Nichol & Dhariwal (a_bar follows a squared
cosine, betas derived from consecutive ratios and clamped to 0.999). All derived
tensors (alphas, a_bar, sqrt terms, posterior variance) are precomputed as [T]
buffers. add_noise() is the closed-form forward jump above. step() is one reverse
update:
    mean = 1/sqrt(a_t) * (x_t - beta_t / sqrt(1 - a_bar_t) * eps_theta)
    x_{t-1} = mean + sqrt(posterior_var_t) * z,  z ~ N(0, I)   (no noise at t=0),
with posterior_var_t = beta_t * (1 - a_bar_{t-1}) / (1 - a_bar_t).

Deliberately simplified vs the DDPM paper and successors: the reverse variance is
fixed (not learned, as in Improved DDPM), sampling is full T-step ancestral only —
no DDIM, no strided/respaced schedules — and step() takes a scalar t, processing
one timestep for the whole batch.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F


class NoiseScheduler:
    def __init__(
        self,
        timesteps: int = 1000,
        schedule: Literal["linear", "cosine"] = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.timesteps = timesteps
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == "cosine":
            steps = timesteps + 1
            t = torch.linspace(0, timesteps, steps) / timesteps
            alpha_bar = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = betas.clamp(0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def add_noise(
        self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """q(x_t | x_0): forward diffusion process."""
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(x0.device)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(x0.device)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    @torch.no_grad()
    def step(
        self, model_output: torch.Tensor, t: int, x_t: torch.Tensor
    ) -> torch.Tensor:
        """One DDPM reverse step: p(x_{t-1} | x_t)."""
        betas_t = self.betas[t].to(x_t.device)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].to(x_t.device)
        sqrt_recip_alpha = (1.0 / self.alphas[t].sqrt()).to(x_t.device)

        mean = sqrt_recip_alpha * (x_t - betas_t / sqrt_one_minus * model_output)
        if t == 0:
            return mean
        posterior_var = self.posterior_variance[t].to(x_t.device)
        noise = torch.randn_like(x_t)
        return mean + posterior_var.sqrt() * noise

    def to(self, device: torch.device) -> "NoiseScheduler":
        for attr in [
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod", "posterior_variance"
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self
