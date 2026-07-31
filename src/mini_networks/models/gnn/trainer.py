"""GCN trainer: full-batch transductive loop + feature-only baseline evidence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataset
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.gnn.config import GNNConfig
from mini_networks.models.gnn.model import GCN

import logging

log = logging.getLogger(__name__)


def _unpack(batch, device):
    """DataLoader(batch_size=1) adds a leading dim — squeeze it back off."""
    x, a, y, train_mask, test_mask = (t.squeeze(0).to(device) for t in batch)
    return x, a, y, train_mask.bool(), test_mask.bool()


class GNNTrainer(BaseTrainer):
    def __init__(self):
        self.model: GCN | None = None

    def _build(self, config: GNNConfig) -> GCN:
        return GCN(
            n_features=config.n_features,
            hidden_dim=config.hidden_dim,
            n_classes=config.n_communities,
            dropout=config.dropout,
        ).to(config.device)

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, GNNConfig)
        model = self._build(config)
        self.model = model
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                     weight_decay=5e-4)
        logger.log_config(config.model_dump())

        batch = next(iter(dataloader))
        x, a, y, train_mask, _ = _unpack(batch, config.device)
        # one epoch == one full-batch step on the length-1 graph dataset
        for epoch in range(config.effective_epochs):
            model.train()
            logits = model(x, a)
            loss = F.cross_entropy(logits[train_mask], y[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0 or epoch == config.effective_epochs - 1:
                train_acc = (logits[train_mask].argmax(-1) == y[train_mask]).float().mean()
                logger.log_metrics(epoch, {
                    "loss": loss.item(), "train_accuracy": train_acc.item(),
                })
        torch.save(model.state_dict(), logger.artifact_path("model.pt"))

    def _mlp_baseline(self, x, y, train_mask, test_mask) -> float:
        """Feature-only logistic regression — the degenerate solution the gate
        must beat. Logged every eval as threshold-honesty evidence."""
        torch.manual_seed(0)
        probe = torch.nn.Linear(x.shape[1], int(y.max().item()) + 1).to(x.device)
        opt = torch.optim.Adam(probe.parameters(), lr=0.05)
        for _ in range(200):
            loss = F.cross_entropy(probe(x[train_mask]), y[train_mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return (probe(x[test_mask]).argmax(-1) == y[test_mask]).float().mean().item()

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        assert isinstance(config, GNNConfig)
        if self.model is None:
            self.model = self._build(config)
        x, a, y, train_mask, test_mask = _unpack(next(iter(dataloader)), config.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x, a)
            acc = (logits[test_mask].argmax(-1) == y[test_mask]).float().mean().item()
        return {
            "accuracy": acc,
            "mlp_baseline_accuracy": self._mlp_baseline(x, y, train_mask, test_mask),
        }

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, GNNConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        node = int(inputs.get("node_id", 0)) if isinstance(inputs, dict) else 0
        ds = _graph_dataset(config)
        x, a, y, _, _ = ds[0]
        node = node % x.shape[0]
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x.to(config.device), a.to(config.device)).argmax(-1)
        n_neighbors = int(ds.adjacency[node].sum().item())
        return {
            "node_id": node,
            "predicted": int(pred[node].item()),
            "true": int(y[node].item()),
            "n_neighbors": n_neighbors,
        }

    def load_checkpoint(self, config: BaseConfig, artifacts_dir) -> None:
        assert isinstance(config, GNNConfig)
        state = torch.load(
            Path(artifacts_dir) / "model.pt", map_location=config.device, weights_only=True
        )
        self.model = self._build(config)
        self.model.load_state_dict(state)
        self.model.eval()


def _graph_dataset(config: GNNConfig):
    return get_dataset(
        config.dataset,
        config.data_root,
        split="train",
        n_nodes=config.n_nodes,
        n_communities=config.n_communities,
        p_in=config.p_in,
        p_out=config.p_out,
        n_features=config.n_features,
        feat_signal=config.feat_signal,
        train_per_class=config.train_per_class,
        seed=config.seed,
    )


def make_gnn_dataloader(config: GNNConfig, split: str = "train") -> DataLoader:
    # batch_size hardcoded to 1 and no sample_limit: the dataset IS one graph;
    # tier budgets act through epochs (1 epoch = 1 full-batch step)
    return DataLoader(_graph_dataset(config), batch_size=1, shuffle=False)
