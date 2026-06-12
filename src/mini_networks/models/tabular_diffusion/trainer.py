"""Tabular diffusion trainer."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.core.diffusion.sampling import sample_loop
from mini_networks.models.tabular_diffusion.config import TabularDiffusionConfig
from mini_networks.models.tabular_diffusion.model import TabularDenoiser


class TabularNoiseScheduler:
    def __init__(self, timesteps: int, beta_start: float, beta_end: float):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.alphas_cumprod[t].sqrt().view(-1, 1).to(x0.device)
        sqrt_one_minus = (1.0 - self.alphas_cumprod[t]).sqrt().view(-1, 1).to(x0.device)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    @torch.no_grad()
    def step(self, model_out: torch.Tensor, t: int, x_t: torch.Tensor) -> torch.Tensor:
        # Simple ancestral step (not exact, but fine for toy)
        if t == 0:
            return x_t - model_out
        return x_t - model_out * 0.5


class TabularDiffusionTrainer(BaseTrainer):
    def __init__(self):
        self.model: TabularDenoiser | None = None
        self.scheduler: TabularNoiseScheduler | None = None

    def _build(self, config: TabularDiffusionConfig) -> TabularDenoiser:
        return TabularDenoiser(n_features=config.n_features, hidden_dim=config.hidden_dim).to(config.device)

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, TabularDiffusionConfig)
        model = self._build(config)
        self.model = model
        scheduler = TabularNoiseScheduler(config.effective_timesteps, config.beta_start, config.beta_end)
        self.scheduler = scheduler
        opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        logger.log_config(config.model_dump())

        for epoch in range(config.effective_epochs):
            model.train()
            total = 0.0
            for x, _ in dataloader:
                x = x.to(config.device)
                B = x.size(0)
                t = torch.randint(0, config.effective_timesteps, (B,), device=config.device)
                noise = torch.randn_like(x)
                xt = scheduler.add_noise(x, noise, t)
                pred = model(xt, t)
                loss = F.mse_loss(pred, noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            logger.log_metrics(epoch, {"loss": total / max(1, len(dataloader)), "epoch": epoch})

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        assert isinstance(config, TabularDiffusionConfig)
        if self.model is None:
            self.model = self._build(config)
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(config.device)
                B = x.size(0)
                t = torch.randint(0, config.effective_timesteps, (B,), device=config.device)
                noise = torch.randn_like(x)
                xt = self.scheduler.add_noise(x, noise, t) if self.scheduler else x
                pred = self.model(xt, t)
                total += F.mse_loss(pred, noise).item()
        return {"eval_loss": total / max(1, len(dataloader))}

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, TabularDiffusionConfig)
        if self.model is None or self.scheduler is None:
            raise RuntimeError("Train first.")
        n = int(inputs.get("n_samples", 4)) if isinstance(inputs, dict) else 4
        x = sample_loop(
            scheduler=self.scheduler,
            predict_noise=lambda x, t_batch, t, _: self.model(x, t_batch),
            shape=(n, config.n_features),
            device=config.device,
            timesteps=config.effective_timesteps,
        )
        return {"samples": x.cpu()}


def make_tabular_diffusion_dataloader(config: TabularDiffusionConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        batch_size=config.effective_batch_size,
        fast_demo=config.effective_fast_demo,
        sample_limit=config.dataset_sample_limit,
        n_features=config.n_features,
        require_downloads=config.require_downloads,
    )
