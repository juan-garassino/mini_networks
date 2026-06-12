"""Resnet trainer."""
from __future__ import annotations

from mini_networks.core.data.registry import make_classification_dataloader
from mini_networks.core.runtime import SupervisedTrainer
from mini_networks.models.resnet.config import ResNetConfig
from mini_networks.models.resnet.model import MiniResNet


class ResNetTrainer(SupervisedTrainer):
    def __init__(self):
        self.model: MiniResNet | None = None

    def _build(self, config: ResNetConfig) -> MiniResNet:
        return MiniResNet(base_channels=config.base_channels, num_classes=config.num_classes).to(config.device)


make_resnet_dataloader = make_classification_dataloader
