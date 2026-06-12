"""Vit trainer."""
from __future__ import annotations

from mini_networks.core.data.registry import make_classification_dataloader
from mini_networks.core.runtime import SupervisedTrainer
from mini_networks.models.vit.config import ViTConfig
from mini_networks.models.vit.model import MiniViT


class ViTTrainer(SupervisedTrainer):
    def __init__(self):
        self.model: MiniViT | None = None

    def _build(self, config: ViTConfig) -> MiniViT:
        return MiniViT(
            patch_size=config.patch_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            mlp_dim=config.mlp_dim,
            num_classes=config.num_classes,
        ).to(config.device)


make_vit_dataloader = make_classification_dataloader
