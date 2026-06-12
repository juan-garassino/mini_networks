"""Classifier trainer."""
from __future__ import annotations

from mini_networks.core.data.registry import make_classification_dataloader
from mini_networks.core.runtime import SupervisedTrainer
from mini_networks.models.classifier.config import ClassifierConfig
from mini_networks.models.classifier.model import SmallCNN


class ClassifierTrainer(SupervisedTrainer):
    def __init__(self):
        self.model: SmallCNN | None = None

    def _build(self, config: ClassifierConfig) -> SmallCNN:
        return SmallCNN(hidden_dim=config.hidden_dim, num_classes=config.num_classes).to(config.device)


make_classifier_dataloader = make_classification_dataloader
