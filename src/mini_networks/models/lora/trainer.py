"""LoRA trainer — two-stage: pre-train on MNIST, fine-tune on FashionMNIST."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.lora.config import LoRAConfig
from mini_networks.models.lora.model import LoRACNN


class LoRATrainer(BaseTrainer):
    def __init__(self):
        self.model: LoRACNN | None = None

    def _build(self, config: LoRAConfig) -> LoRACNN:
        return LoRACNN(
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
        ).to(config.device)

    def _run_epoch(
        self,
        model: LoRACNN,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        device: str,
    ) -> float:
        model.train()
        total_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(1, len(dataloader))

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, LoRAConfig)
        model = self._build(config)
        self.model = model
        logger.log_config(config.model_dump())

        # --- Stage 1: pre-train on MNIST (full model, all params trainable) ---
        print("  [LoRA] Stage 1: pre-training on MNIST")
        model.unfreeze_all()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        mnist_dl = make_lora_dataloader(config, dataset="mnist", split="train")
        for epoch in range(config.tier_epochs(config.pretrain_epochs, medium_cap=2)):
            avg = self._run_epoch(model, mnist_dl, optimizer, config.device)
            logger.log_metrics(epoch, {"pretrain_loss": avg, "stage": 1, "epoch": epoch})
            print(f"    pretrain epoch {epoch}  loss {avg:.4f}")

        # --- Stage 2: fine-tune on FashionMNIST (only LoRA params) ---
        print("  [LoRA] Stage 2: fine-tuning on FashionMNIST with LoRA")
        model.freeze_for_finetune(freeze_conv=config.freeze_conv)
        lora_optimizer = optim.Adam(model.trainable_params(), lr=config.learning_rate)

        fashion_dl = make_lora_dataloader(config, dataset="fashion_mnist", split="train")
        offset = config.tier_epochs(config.pretrain_epochs, medium_cap=2)
        for epoch in range(config.tier_epochs(config.finetune_epochs, medium_cap=2)):
            avg = self._run_epoch(model, fashion_dl, lora_optimizer, config.device)
            logger.log_metrics(offset + epoch, {"finetune_loss": avg, "stage": 2, "epoch": epoch})
            print(f"    finetune epoch {epoch}  loss {avg:.4f}")

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        assert isinstance(config, LoRAConfig)
        if self.model is None:
            self.model = self._build(config)
        model = self.model
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(config.device), labels.to(config.device)
                logits = model(images)
                total_loss += F.cross_entropy(logits, labels).item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return {
            "eval_loss": total_loss / max(1, len(dataloader)),
            "accuracy": correct / max(1, total),
        }

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, LoRAConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        model = self.model
        model.eval()
        # inputs: {"image": tensor [1,1,28,28]} or {"images": tensor [B,1,28,28]}
        if isinstance(inputs, dict):
            x = inputs.get("image") if "image" in inputs else inputs.get("images")
        else:
            x = inputs
        x = torch.as_tensor(x, dtype=torch.float32).to(config.device)
        with torch.no_grad():
            logits = model(x)
            preds = logits.argmax(dim=-1)
        return {"predictions": preds.cpu().tolist(), "logits": logits.cpu().tolist()}


def make_lora_dataloader(
    config: LoRAConfig,
    dataset: str = "mnist",
    split: str = "train",
) -> DataLoader:
    return get_dataloader(
        name=dataset,
        data_root=config.data_root,
        split=split,
        task="classification",
        batch_size=config.effective_batch_size,
        fast_demo=config.effective_fast_demo,
        sample_limit=config.dataset_sample_limit,
    )
