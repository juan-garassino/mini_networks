"""DINO trainer: contrastive-loader views, self-distillation loss, EMA teacher."""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.runtime import ContrastiveTrainer
from mini_networks.models.dino.config import DINOConfig
from mini_networks.models.dino.model import MiniDINO


class DINOTrainer(ContrastiveTrainer):
    def __init__(self):
        self.model: MiniDINO | None = None

    def _build(self, config: DINOConfig) -> MiniDINO:
        return MiniDINO(
            patch_size=config.patch_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            mlp_dim=config.mlp_dim,
            proj_hidden=config.proj_hidden,
            out_dim=config.out_dim,
            student_temp=config.student_temp,
            teacher_temp=config.teacher_temp,
            ema_decay=config.ema_decay,
            center_momentum=config.center_momentum,
        ).to(config.device)

    def _optimizer(self, model: MiniDINO, config: DINOConfig):
        # Only the student gets gradients; the teacher moves by EMA in _post_step.
        return torch.optim.Adam(model._student_params(), lr=config.learning_rate)

    def _forward(self, model: MiniDINO, batch, config: DINOConfig):
        v1, v2, _ = batch
        return model.forward_views(v1.to(config.device), v2.to(config.device))

    def _loss(self, emb_a: torch.Tensor, emb_b: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        # Temperatures live in the module (student 0.1 / teacher 0.04); the
        # base-class `temperature` kwarg is ignored.
        return self.model.dino_loss(emb_a, emb_b)

    def _post_step(self, model: MiniDINO) -> None:
        model.update_teacher()

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, DINOConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        model = self.model
        model.eval()
        with torch.no_grad():
            images = inputs if not isinstance(inputs, dict) else inputs.get("images")
            embeds = model.embed(images.to(config.device))
        return {"embeddings": embeds.cpu()}


def make_dino_dataloader(config: DINOConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        task="contrastive",
        batch_size=config.effective_batch_size,
        fast_demo=config.effective_fast_demo,
        sample_limit=config.dataset_sample_limit,
        image_size=28,
    )
