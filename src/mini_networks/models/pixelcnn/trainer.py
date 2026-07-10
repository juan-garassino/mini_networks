"""PixelCNN trainer."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.pixelcnn.config import PixelCNNConfig
from mini_networks.models.pixelcnn.model import PixelCNN


class PixelCNNTrainer(BaseTrainer):
    def __init__(self):
        self.model: PixelCNN | None = None

    def _build(self, config: PixelCNNConfig) -> PixelCNN:
        return PixelCNN(n_filters=config.n_filters, n_layers=config.n_layers).to(config.device)

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, PixelCNNConfig)
        model = self._build(config)
        self.model = model
        opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        logger.log_config(config.model_dump())

        for epoch in range(config.effective_epochs):
            model.train()
            total = 0.0
            for images, _ in dataloader:
                # Binarised MNIST: the model is an autoregressive Bernoulli
                # p(x_i=1 | x_<i), trained with per-pixel BCE — not a
                # reconstruction MSE (the masks keep the target pixel unseen).
                binary = (images.to(config.device) > 0.5).float()
                logits = model(binary)
                loss = F.binary_cross_entropy_with_logits(logits, binary)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            logger.log_metrics(epoch, {"loss": total / max(1, len(dataloader))})

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        assert isinstance(config, PixelCNNConfig)
        if self.model is None:
            self.model = self._build(config)
        model = self.model
        model.eval()
        total = 0.0
        with torch.no_grad():
            for images, _ in dataloader:
                binary = (images.to(config.device) > 0.5).float()
                logits = model(binary)
                total += F.binary_cross_entropy_with_logits(logits, binary).item()
        return {"eval_loss": total / max(1, len(dataloader))}

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, PixelCNNConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        model = self.model
        model.eval()
        n = int(inputs.get("n_samples", 4)) if isinstance(inputs, dict) else 4
        seed = inputs.get("seed") if isinstance(inputs, dict) else None
        if seed is not None:
            torch.manual_seed(int(seed))
        # True raster-scan sampling: one forward pass per pixel, each drawn
        # from its Bernoulli conditional on everything above/left. Sequential
        # by construction — the parallel trick only exists in training.
        x = torch.zeros(n, 1, 28, 28, device=config.device)
        with torch.no_grad():
            for i in range(28):
                for j in range(28):
                    probs = torch.sigmoid(model(x)[:, :, i, j])
                    x[:, :, i, j] = torch.bernoulli(probs)
        return {"samples": x.cpu()}


def make_pixelcnn_dataloader(config: PixelCNNConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        task="classification",
        batch_size=config.effective_batch_size,
        fast_demo=config.effective_fast_demo,
        sample_limit=config.dataset_sample_limit,
    )
