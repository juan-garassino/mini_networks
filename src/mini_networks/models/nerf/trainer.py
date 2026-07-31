"""Mini-NeRF trainer: ray-pool batches, PSNR on held-out azimuths."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.nerf.config import NerfConfig
from mini_networks.models.nerf.model import MiniNeRF, render_rays
from mini_networks.models.nerf.scene import make_nerf_dataloader, make_rays

import logging

log = logging.getLogger(__name__)


class NerfTrainer(BaseTrainer):
    def __init__(self):
        self.model: MiniNeRF | None = None

    def _build(self, config: NerfConfig) -> MiniNeRF:
        return MiniNeRF(
            pe_freqs=config.pe_freqs,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
        ).to(config.device)

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, NerfConfig)
        model = self._build(config)
        self.model = model
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        logger.log_config(config.model_dump())

        # flatten all training views into one ray pool
        pool = []
        for rays_o, rays_d, colors in dataloader:
            pool.append(torch.cat([rays_o[0], rays_d[0], colors[0]], dim=-1))
        pool = torch.cat(pool, dim=0).to(config.device)     # [N, 9]
        n_samples = config.effective_timesteps
        g = torch.Generator(device="cpu").manual_seed(config.seed)

        for epoch in range(config.effective_epochs):
            model.train()
            perm = torch.randperm(pool.shape[0], generator=g).to(config.device)
            total, n_batches = 0.0, 0
            max_batches = config.max_train_batches or (pool.shape[0] // config.rays_per_batch + 1)
            for i in range(0, pool.shape[0], config.rays_per_batch):
                if n_batches >= max_batches:
                    break
                batch = pool[perm[i:i + config.rays_per_batch]]
                rgb = render_rays(model, batch[:, :3], batch[:, 3:6], n_samples,
                                  config.near, config.far, stratified=True)
                loss = F.mse_loss(rgb, batch[:, 6:])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
                n_batches += 1
            logger.log_metrics(epoch, {"loss": total / max(1, n_batches), "epoch": epoch})
            log.info(f"  epoch {epoch}  loss {total / max(1, n_batches):.5f}")

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))

    @torch.no_grad()
    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        """PSNR on held-out azimuths, rendered with deterministic bin midpoints."""
        assert isinstance(config, NerfConfig)
        if self.model is None:
            self.model = self._build(config)
        model = self.model
        model.eval()
        test_dl = make_nerf_dataloader(config, split="test")
        psnrs = []
        for rays_o, rays_d, colors in test_dl:
            rgb = render_rays(model, rays_o[0].to(config.device),
                              rays_d[0].to(config.device),
                              config.effective_timesteps, config.near, config.far,
                              stratified=False)
            mse = F.mse_loss(rgb, colors[0].to(config.device)).item()
            psnrs.append(-10.0 * torch.log10(torch.tensor(mse + 1e-10)).item())
        return {
            "psnr": sum(psnrs) / max(1, len(psnrs)),
            "psnr_min": min(psnrs) if psnrs else 0.0,
        }

    @torch.no_grad()
    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, NerfConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        azimuth = float(inputs.get("azimuth", 30.0)) if isinstance(inputs, dict) else 30.0
        rays_o, rays_d = make_rays(azimuth, config.elevation_deg,
                                   config.image_size, config.radius)
        self.model.eval()
        rgb = render_rays(self.model, rays_o.to(config.device), rays_d.to(config.device),
                          config.effective_timesteps, config.near, config.far,
                          stratified=False)
        img = rgb.T.reshape(3, config.image_size, config.image_size).cpu()
        return {"image": img, "azimuth": azimuth}

    def load_checkpoint(self, config: BaseConfig, artifacts_dir) -> None:
        assert isinstance(config, NerfConfig)
        state = torch.load(
            Path(artifacts_dir) / "model.pt", map_location=config.device, weights_only=True
        )
        self.model = self._build(config)
        self.model.load_state_dict(state)
        self.model.eval()
