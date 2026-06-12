"""Convnext trainer."""
from __future__ import annotations

from mini_networks.core.data.registry import make_classification_dataloader
from mini_networks.core.runtime import SupervisedTrainer
from mini_networks.models.convnext.config import ConvNeXtConfig
from mini_networks.models.convnext.model import TinyConvNeXt


class ConvNeXtTrainer(SupervisedTrainer):
    def __init__(self):
        self.model: TinyConvNeXt | None = None

    def _build(self, config: ConvNeXtConfig) -> TinyConvNeXt:
        return TinyConvNeXt(base_channels=config.base_channels, num_classes=config.num_classes).to(config.device)


make_convnext_dataloader = make_classification_dataloader
