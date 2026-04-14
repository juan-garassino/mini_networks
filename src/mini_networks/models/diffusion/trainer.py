"""DDPM trainer with EMA, curriculum learning, and LR warmup."""
from __future__ import annotations

import copy
from typing import Any

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.diffusion.config import DiffusionConfig
from mini_networks.models.diffusion.model import UNet
from mini_networks.models.diffusion.scheduler import NoiseScheduler
from mini_networks.core.diffusion.sampling import sample_loop


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

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
      # at inference:
      with ema.average_parameters():
          samples = model(...)
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

    def state_dict(self) -> dict:
        return self.shadow.state_dict()


# ---------------------------------------------------------------------------
# Curriculum helpers
# ---------------------------------------------------------------------------

def _image_complexity(images: torch.Tensor) -> torch.Tensor:
    """Pixel-variance as a proxy for image complexity. Shape: [B]."""
    return images.view(images.size(0), -1).var(dim=1)


def _sort_batch_by_complexity(images: torch.Tensor, labels: torch.Tensor, descending: bool = True):
    """Sort a batch from most to least complex (hard-first curriculum)."""
    complexity = _image_complexity(images)
    order = complexity.argsort(descending=descending)
    return images[order], labels[order]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DDPMTrainer(BaseTrainer):
    def __init__(self):
        self.model: UNet | None = None
        self.scheduler: NoiseScheduler | None = None
        self.ema: EMA | None = None

    def _build(self, config: DiffusionConfig):
        model = UNet(
            in_channels=config.in_channels,
            base_channels=config.base_channels,
        ).to(config.device)
        scheduler = NoiseScheduler(
            timesteps=config.timesteps,
            schedule=config.schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        ).to(torch.device(config.device))
        return model, scheduler

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, DiffusionConfig)
        model, scheduler = self._build(config)
        self.model = model
        self.scheduler = scheduler
        logger.log_config(config.model_dump())

        # EMA
        use_ema = config.ema_decay > 0.0
        if use_ema:
            ema = EMA(model, decay=config.ema_decay)
            self.ema = ema

        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # LR warmup via linear scheduler
        if config.warmup_steps > 0:
            def lr_lambda(step: int) -> float:
                return min(1.0, step / config.warmup_steps)
            scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler_lr = None

        T = config.timesteps
        global_step = 0

        for epoch in range(config.effective_epochs):
            model.train()
            total_loss = 0.0
            for images, labels in dataloader:
                images = images.to(config.device) * 2.0 - 1.0  # [-1, 1]
                labels = labels.to(config.device)

                # Curriculum: sort batch from hardest to easiest
                if config.curriculum:
                    images, labels = _sort_batch_by_complexity(images, labels)

                B = images.shape[0]
                t = torch.randint(0, T, (B,), device=config.device)
                noise = torch.randn_like(images)
                noisy = scheduler.add_noise(images, noise, t)
                pred_noise = model(noisy, t)
                loss = F.mse_loss(pred_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if scheduler_lr is not None:
                    scheduler_lr.step()

                if use_ema:
                    ema.update(model)

                total_loss += loss.item()
                global_step += 1

            avg = total_loss / max(1, len(dataloader))
            logger.log_metrics(epoch, {"loss": avg, "epoch": epoch})

        # Save live model
        torch.save(model.state_dict(), logger.artifact_path("model.pt"))
        # Save EMA model separately if used
        if use_ema:
            torch.save(ema.state_dict(), logger.artifact_path("model_ema.pt"))

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        assert isinstance(config, DiffusionConfig)
        if self.model is None:
            self.model, self.scheduler = self._build(config)
        model = self.model
        scheduler = self.scheduler
        model.eval()
        T = config.timesteps
        total_loss = 0.0
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(config.device) * 2.0 - 1.0
                B = images.shape[0]
                t = torch.randint(0, T, (B,), device=config.device)
                noise = torch.randn_like(images)
                noisy = scheduler.add_noise(images, noise, t)
                pred_noise = model(noisy, t)
                total_loss += F.mse_loss(pred_noise, noise).item()
        return {"eval_loss": total_loss / max(1, len(dataloader))}

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, DiffusionConfig)
        if self.model is None or self.scheduler is None:
            self.model, self.scheduler = self._build(config)
        # Use EMA model for inference if available
        model = self.ema.shadow if self.ema is not None else self.model
        scheduler = self.scheduler
        model.eval()
        n_samples = inputs.get("n_samples", 4) if isinstance(inputs, dict) else 4
        shape = (n_samples, config.in_channels, config.image_size, config.image_size)
        with torch.no_grad():
            x = sample_loop(
                scheduler=scheduler,
                predict_noise=lambda x, t_batch, t, _: model(x, t_batch),
                shape=shape,
                device=config.device,
                timesteps=config.timesteps,
            )
        samples = (x.clamp(-1, 1) + 1) / 2
        return {"samples": samples.cpu()}


    def load_checkpoint(self, config: BaseConfig, artifacts_dir) -> None:
        """Prefer model_ema.pt when present, then model.pt. Also rebuilds scheduler."""
        from pathlib import Path
        assert isinstance(config, DiffusionConfig)
        path = Path(artifacts_dir)
        model, scheduler = self._build(config)
        self.scheduler = scheduler
        ema_path = path / "model_ema.pt"
        load_path = ema_path if ema_path.exists() else path / "model.pt"
        state = torch.load(load_path, map_location=config.device)
        model.load_state_dict(state)
        model.eval()
        self.model = model
        self.ema = None  # weights already loaded directly into self.model


def make_diffusion_dataloader(config: DiffusionConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        task="classification",
        batch_size=config.effective_batch_size,
        fast_demo=config.effective_fast_demo,
        sample_limit=config.dataset_sample_limit,
    )
