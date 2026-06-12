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
        state = torch.load(path / "model.pt", map_location=config.device, weights_only=True)
        if self.model is None:
            self.model = self._build(config)
        self.model.load_state_dict(state)
        self.model.eval()

    def load_training_state(self, config: BaseConfig, logger: Logger) -> dict[str, Any] | None:
        """Load resumable training state when available."""
        return logger.load_training_state()

    def _iter_train_batches(self, dataloader: DataLoader, config: BaseConfig):
        limit = config.max_train_batches
        for batch_idx, batch in enumerate(dataloader):
            if limit is not None and batch_idx >= limit:
                break
            yield batch

    def _iter_eval_batches(self, dataloader: DataLoader, config: BaseConfig):
        limit = config.max_eval_batches
        for batch_idx, batch in enumerate(dataloader):
            if limit is not None and batch_idx >= limit:
                break
            yield batch


class SupervisedTrainer(BaseTrainer):
    """Base class with common supervised loops (classification/regression)."""

    def _build(self, config: BaseConfig):
        raise NotImplementedError

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        return F.cross_entropy(logits, targets)

    def _forward(self, model, batch, config: BaseConfig):
        x, y = batch
        return model(x), y

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        model = self._build(config)
        self.model = model
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        logger.log_config(config.model_dump())
        start_epoch = 0

        if config.resume:
            state = self.load_training_state(config, logger)
            if state and not state.get("completed", False):
                model.load_state_dict(state["model_state"])
                optimizer.load_state_dict(state["optimizer_state"])
                start_epoch = int(state.get("epoch", -1)) + 1

        for epoch in range(start_epoch, config.effective_epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            for batch in self._iter_train_batches(dataloader, config):
                logits, targets = self._forward(model, batch, config)
                logits, targets = logits.to(config.device), targets.to(config.device)
                loss = self._loss(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
            logger.log_metrics(epoch, {
                "loss": total_loss / max(1, len(dataloader)),
                "accuracy": correct / max(1, total),
                "epoch": epoch,
            })
            logger.save_training_state({
                "epoch": epoch,
                "completed": False,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            })

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))
        logger.save_training_state({
            "epoch": config.effective_epochs - 1,
            "completed": True,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        })
        logger.log_summary({"status": "completed", "epochs": config.effective_epochs})

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        if self.model is None:
            self.model = self._build(config)
        model = self.model
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self._iter_eval_batches(dataloader, config):
                logits, targets = self._forward(model, batch, config)
                logits, targets = logits.to(config.device), targets.to(config.device)
                total_loss += self._loss(logits, targets).item()
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        return {
            "eval_loss": total_loss / max(1, len(dataloader)),
            "accuracy": correct / max(1, total),
        }


class ContrastiveTrainer(BaseTrainer):
    """Base class for contrastive embedding training."""

    def _build(self, config: BaseConfig):
        raise NotImplementedError

    def _loss(self, emb_a: torch.Tensor, emb_b: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        import torch.nn.functional as F
        logits = (emb_a @ emb_b.T) / temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, targets)

    def _forward(self, model, batch, config: BaseConfig):
        raise NotImplementedError

    def _optimizer(self, model, config: BaseConfig):
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        model = self._build(config)
        self.model = model
        optimizer = self._optimizer(model, config)
        logger.log_config(config.model_dump())
        start_epoch = 0

        if config.resume:
            state = self.load_training_state(config, logger)
            if state and not state.get("completed", False):
                model.load_state_dict(state["model_state"])
                optimizer.load_state_dict(state["optimizer_state"])
                start_epoch = int(state.get("epoch", -1)) + 1

        for epoch in range(start_epoch, config.effective_epochs):
            model.train()
            total = 0.0
            for batch in self._iter_train_batches(dataloader, config):
                emb_a, emb_b = self._forward(model, batch, config)
                loss = self._loss(emb_a, emb_b, temperature=getattr(config, "temperature", 0.2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            logger.log_metrics(epoch, {"loss": total / max(1, len(dataloader)), "epoch": epoch})
            logger.save_training_state({
                "epoch": epoch,
                "completed": False,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            })

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))
        logger.save_training_state({
            "epoch": config.effective_epochs - 1,
            "completed": True,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        })
        logger.log_summary({"status": "completed", "epochs": config.effective_epochs})

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        if self.model is None:
            self.model = self._build(config)
        model = self.model
        model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in self._iter_eval_batches(dataloader, config):
                emb_a, emb_b = self._forward(model, batch, config)
                total += self._loss(emb_a, emb_b, temperature=getattr(config, "temperature", 0.2)).item()
        return {"eval_loss": total / max(1, len(dataloader))}


class SegmentationTrainerBase(BaseTrainer):
    """Base class for segmentation training."""

    def _build(self, config: BaseConfig):
        raise NotImplementedError

    def _forward(self, model, batch, config: BaseConfig):
        raise NotImplementedError

    def _loss(self, preds: torch.Tensor, targets: torch.Tensor, config: BaseConfig) -> torch.Tensor:
        import torch.nn.functional as F
        return F.cross_entropy(preds, targets)

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        model = self._build(config)
        self.model = model
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        logger.log_config(config.model_dump())
        start_epoch = 0

        if config.resume:
            state = self.load_training_state(config, logger)
            if state and not state.get("completed", False):
                model.load_state_dict(state["model_state"])
                optimizer.load_state_dict(state["optimizer_state"])
                start_epoch = int(state.get("epoch", -1)) + 1

        for epoch in range(start_epoch, config.effective_epochs):
            model.train()
            total = 0.0
            for batch in self._iter_train_batches(dataloader, config):
                preds, targets = self._forward(model, batch, config)
                loss = self._loss(preds, targets, config)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            logger.log_metrics(epoch, {"loss": total / max(1, len(dataloader)), "epoch": epoch})
            logger.save_training_state({
                "epoch": epoch,
                "completed": False,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            })

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))
        logger.save_training_state({
            "epoch": config.effective_epochs - 1,
            "completed": True,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        })
        logger.log_summary({"status": "completed", "epochs": config.effective_epochs})

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        if self.model is None:
            self.model = self._build(config)
        model = self.model
        model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in self._iter_eval_batches(dataloader, config):
                preds, targets = self._forward(model, batch, config)
                total += self._loss(preds, targets, config).item()
        return {"eval_loss": total / max(1, len(dataloader))}


class DetectionTrainerBase(BaseTrainer):
    """Base class for detection training (single bbox)."""

    def _build(self, config: BaseConfig):
        raise NotImplementedError

    def _forward(self, model, batch, config: BaseConfig):
        raise NotImplementedError

    def _loss(
        self,
        class_logits: torch.Tensor,
        bbox_pred: torch.Tensor,
        labels: torch.Tensor,
        target_bbox: torch.Tensor,
        config: BaseConfig,
    ) -> torch.Tensor:
        import torch.nn.functional as F
        cls_loss = F.cross_entropy(class_logits, labels)
        bbox_loss = F.mse_loss(bbox_pred, target_bbox)
        return cls_loss + bbox_loss

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        model = self._build(config)
        self.model = model
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        logger.log_config(config.model_dump())
        start_epoch = 0

        if config.resume:
            state = self.load_training_state(config, logger)
            if state and not state.get("completed", False):
                model.load_state_dict(state["model_state"])
                optimizer.load_state_dict(state["optimizer_state"])
                start_epoch = int(state.get("epoch", -1)) + 1

        for epoch in range(start_epoch, config.effective_epochs):
            model.train()
            total = 0.0
            for batch in self._iter_train_batches(dataloader, config):
                class_logits, bbox_pred, labels, target_bbox = self._forward(model, batch, config)
                loss = self._loss(class_logits, bbox_pred, labels, target_bbox, config)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            logger.log_metrics(epoch, {"loss": total / max(1, len(dataloader)), "epoch": epoch})
            logger.save_training_state({
                "epoch": epoch,
                "completed": False,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            })

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))
        logger.save_training_state({
            "epoch": config.effective_epochs - 1,
            "completed": True,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        })
        logger.log_summary({"status": "completed", "epochs": config.effective_epochs})

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        if self.model is None:
            self.model = self._build(config)
        model = self.model
        model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in self._iter_eval_batches(dataloader, config):
                class_logits, bbox_pred, labels, target_bbox = self._forward(model, batch, config)
                total += self._loss(class_logits, bbox_pred, labels, target_bbox, config).item()
        return {"eval_loss": total / max(1, len(dataloader))}
