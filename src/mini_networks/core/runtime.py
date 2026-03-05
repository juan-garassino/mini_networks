"""Runtime protocol: defines the train/evaluate/infer contract."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.logging.logger import Logger


class BaseTrainer(ABC):
    """All model trainers must implement this interface."""

    @abstractmethod
    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        """Run training loop."""

    @abstractmethod
    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        """Run evaluation and return metrics dict."""

    @abstractmethod
    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        """Run inference on inputs, return outputs."""

    def load_checkpoint(self, config: BaseConfig, artifacts_dir: str | Path) -> None:
        """Load model weights from a checkpoint directory.

        Default implementation expects:
          - artifacts_dir/model.pt  — PyTorch state dict
          - self._build(config)     — returns a single nn.Module
          - self.model              — attribute on the trainer

        Override for non-standard layouts (GAN, Diffusion, RL, text models).
        """
        path = Path(artifacts_dir)
        state = torch.load(path / "model.pt", map_location=config.device)
        if self.model is None:
            self.model = self._build(config)
        self.model.load_state_dict(state)
        self.model.eval()
