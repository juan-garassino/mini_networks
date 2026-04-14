"""Tabular classifier trainer."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer, SupervisedTrainer
from mini_networks.core.data.tabular import normalize_batch
from mini_networks.models.tabular.config import TabularClassifierConfig
from mini_networks.models.tabular.model import TabularMLP, TabularLinear, TabularTransformer


class TabularClassifierTrainer(SupervisedTrainer):
    def __init__(self):
        self.model: TabularMLP | None = None

    def _build(self, config: TabularClassifierConfig) -> TabularMLP:
        if config.model_type == "linear":
            return TabularLinear(n_features=config.n_features, n_classes=config.n_classes).to(config.device)
        if config.model_type == "transformer":
            return TabularTransformer(
                n_features=config.n_features,
                n_classes=config.n_classes,
                d_model=config.hidden_dim,
            ).to(config.device)
        return TabularMLP(
            n_features=config.n_features,
            n_classes=config.n_classes,
            hidden=config.hidden_dim,
        ).to(config.device)

    def _forward(self, model, batch, config: TabularClassifierConfig):
        x, y = batch
        x = normalize_batch(x.to(config.device))
        return model(x), y

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, TabularClassifierConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        model = self.model
        model.eval()
        with torch.no_grad():
            feats = inputs if not isinstance(inputs, dict) else inputs.get("features")
            feats = torch.as_tensor(feats, dtype=torch.float32).to(config.device)
            logits = model(feats)
            preds = logits.argmax(dim=-1)
        return {"predictions": preds.cpu().tolist(), "logits": logits.cpu().tolist()}


def make_tabular_dataloader(config: TabularClassifierConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        batch_size=config.effective_batch_size,
        fast_demo=config.effective_fast_demo,
        sample_limit=config.dataset_sample_limit,
        n_features=config.n_features,
        require_downloads=config.require_downloads,
    )
