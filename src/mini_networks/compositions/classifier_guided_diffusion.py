"""Classifier-guided diffusion composition (Dhariwal-style guidance)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.models.classifier.model import SmallCNN
from mini_networks.models.diffusion.model import UNet
from mini_networks.models.diffusion.scheduler import NoiseScheduler
from mini_networks.core.diffusion.sampling import sample_loop


class ClassifierGuidedDiffusionConfig(BaseConfig):
    model_name: str = "classifier_guided_diffusion"

    # Diffusion
    timesteps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02
    base_channels: int = 32

    # Classifier
    cls_hidden: int = 64
    num_classes: int = 10
    classifier_epochs: int = 2

    # Guidance
    guide_scale: float = 2.0
    target_class: int = 0

    # Data
    dataset: str = "mnist"


class ClassifierGuidedDiffusion:
    def __init__(self):
        self.unet: UNet | None = None
        self.classifier: SmallCNN | None = None
        self.scheduler: NoiseScheduler | None = None

    def _build_unet(self, config: ClassifierGuidedDiffusionConfig) -> UNet:
        return UNet(in_channels=1, base_channels=config.base_channels).to(config.device)

    def _build_classifier(self, config: ClassifierGuidedDiffusionConfig) -> SmallCNN:
        return SmallCNN(hidden_dim=config.cls_hidden, num_classes=config.num_classes).to(config.device)

    def train_classifier(self, config: ClassifierGuidedDiffusionConfig, logger: Logger) -> None:
        dl = get_dataloader(
            name=config.dataset,
            data_root=config.data_root,
            split="train",
            task="classification",
            batch_size=config.effective_batch_size,
            fast_demo=config.effective_fast_demo,
            sample_limit=config.dataset_sample_limit,
        )
        clf = self._build_classifier(config)
        opt = torch.optim.Adam(clf.parameters(), lr=config.learning_rate)
        epochs = config.tier_epochs(config.classifier_epochs, medium_cap=2)
        for epoch in range(epochs):
            clf.train()
            total = 0.0
            for images, labels in dl:
                images, labels = images.to(config.device), labels.to(config.device)
                logits = clf(images)
                loss = F.cross_entropy(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            logger.log_metrics(epoch, {"cls_loss": total / max(1, len(dl))})
        self.classifier = clf
        torch.save(clf.state_dict(), logger.artifact_path("classifier.pt"))

    def train_diffusion(self, config: ClassifierGuidedDiffusionConfig, logger: Logger) -> None:
        dl = get_dataloader(
            name=config.dataset,
            data_root=config.data_root,
            split="train",
            task="classification",
            batch_size=config.effective_batch_size,
            fast_demo=config.effective_fast_demo,
            sample_limit=config.dataset_sample_limit,
        )
        unet = self._build_unet(config)
        scheduler = NoiseScheduler(
            timesteps=config.effective_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        ).to(torch.device(config.device))
        opt = torch.optim.Adam(unet.parameters(), lr=config.learning_rate)
        epochs = config.effective_epochs
        for epoch in range(epochs):
            unet.train()
            total = 0.0
            for images, _ in dl:
                images = images.to(config.device) * 2.0 - 1.0
                B = images.shape[0]
                t = torch.randint(0, config.effective_timesteps, (B,), device=config.device)
                noise = torch.randn_like(images)
                xt = scheduler.add_noise(images, noise, t)
                pred = unet(xt, t)
                loss = F.mse_loss(pred, noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            logger.log_metrics(epoch, {"diff_loss": total / max(1, len(dl))})
        self.unet = unet
        self.scheduler = scheduler
        torch.save(unet.state_dict(), logger.artifact_path("unet.pt"))

    def sample(self, config: ClassifierGuidedDiffusionConfig, n: int = 4) -> torch.Tensor:
        if self.unet is None or self.classifier is None or self.scheduler is None:
            raise RuntimeError("Train classifier and diffusion before sampling.")
        unet = self.unet
        clf = self.classifier
        sched = self.scheduler
        unet.eval()
        clf.eval()

        target = torch.full((n,), int(config.target_class), device=config.device, dtype=torch.long)

        def predict_noise(x, t_batch, t, _):
            return unet(x, t_batch)

        def guidance(x, t_batch, t, eps, _):
            x_in = x.detach().requires_grad_(True)
            logits = clf((x_in + 1.0) / 2.0)
            logp = F.log_softmax(logits, dim=-1)
            score = logp[torch.arange(x_in.size(0), device=config.device), target].sum()
            grad = torch.autograd.grad(score, x_in)[0]
            return eps - config.guide_scale * grad

        x = sample_loop(
            scheduler=sched,
            predict_noise=predict_noise,
            guidance_fn=guidance,
            shape=(n, 1, 28, 28),
            device=config.device,
            timesteps=config.effective_timesteps,
        )

        return (x + 1.0) / 2.0

    def run(self, config: ClassifierGuidedDiffusionConfig, logger: Logger) -> None:
        self.train_classifier(config, logger)
        self.train_diffusion(config, logger)
