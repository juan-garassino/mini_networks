"""Shared composition bases for multimodal pipelines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.logging.logger import Logger
from mini_networks.models.clip.data import label_to_tokens


class CompositionBase(ABC):
    """Base class for compositions with a unified infer signature."""

    @abstractmethod
    def infer(self, config: BaseConfig, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run inference. Inputs must be a dict."""

    def _as_tensor(self, inputs: dict[str, Any], key: str, dtype=torch.float32) -> torch.Tensor:
        if key not in inputs:
            raise KeyError(f"Missing required input key: {key}")
        return torch.as_tensor(inputs[key], dtype=dtype)


class ContrastiveCompositionBase(CompositionBase):
    """Base class for contrastive dual-modality compositions."""

    def __init__(self):
        self.modules: dict[str, nn.Module] = {}

    @abstractmethod
    def _build_modules(self, config: BaseConfig) -> dict[str, nn.Module]:
        """Return a dict of modules to be trained."""

    @abstractmethod
    def _get_dataloader(self, config: BaseConfig) -> DataLoader:
        """Return a training dataloader."""

    @abstractmethod
    def _encode_pair(
        self,
        modules: dict[str, nn.Module],
        primary: torch.Tensor,
        tokens: torch.Tensor,
        config: BaseConfig,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return embeddings for contrastive loss."""

    @abstractmethod
    def _infer_embeddings(self, config: BaseConfig, inputs: dict[str, Any]) -> torch.Tensor:
        """Return embeddings for inference."""

    def _prepare_tokens(self, labels: torch.Tensor, config: BaseConfig) -> torch.Tensor:
        seq_len = getattr(config, "text_seq_len", 32)
        vocab_size = getattr(config, "vocab_size", 256)
        return torch.stack(
            [label_to_tokens(int(l), seq_len, vocab_size) for l in labels], dim=0
        )

    def _contrastive_logits(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor | None,
        temperature: float,
    ) -> torch.Tensor:
        if emb_b is None:
            return (emb_a @ emb_a.T) / temperature
        return (emb_a @ emb_b.T) / temperature

    def _optimizer(self, modules: dict[str, nn.Module], config: BaseConfig):
        params = []
        for module in modules.values():
            params.extend(list(module.parameters()))
        return torch.optim.Adam(params, lr=config.learning_rate)

    def train(self, config: BaseConfig, logger: Logger) -> None:
        modules = self._build_modules(config)
        self.modules = modules
        opt = self._optimizer(modules, config)
        dl = self._get_dataloader(config)
        temperature = getattr(config, "temperature", 0.2)

        for epoch in range(config.effective_epochs):
            total = 0.0
            for primary, labels in dl:
                primary = primary.to(config.device)
                tokens = self._prepare_tokens(labels, config).to(config.device)
                out = self._encode_pair(modules, primary, tokens, config)
                if isinstance(out, tuple):
                    emb_a, emb_b = out
                else:
                    emb_a, emb_b = out, None
                logits = self._contrastive_logits(emb_a, emb_b, temperature)
                targets = torch.arange(logits.size(0), device=logits.device)
                loss = F.cross_entropy(logits, targets)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            logger.log_metrics(epoch, {"loss": total / max(1, len(dl))})

        # Save modules
        if "model" in modules and len(modules) == 1:
            torch.save(modules["model"].state_dict(), logger.artifact_path("model.pt"))
        else:
            for name, module in modules.items():
                torch.save(module.state_dict(), logger.artifact_path(f"{name}.pt"))

    def infer(self, config: BaseConfig, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.modules:
            raise RuntimeError("Train first.")
        emb = self._infer_embeddings(config, inputs)
        return {"embeddings": emb.cpu()}
