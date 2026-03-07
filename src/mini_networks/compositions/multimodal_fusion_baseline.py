"""Multimodal fusion baseline: image + text -> class."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.models.multimodal.blocks import MultiModalEncoder
from mini_networks.models.clip.data import label_to_tokens


class MultimodalFusionConfig(BaseConfig):
    model_name: str = "multimodal_fusion_baseline"
    d_model: int = 64
    vocab_size: int = 256
    text_seq_len: int = 32
    num_classes: int = 10
    fusion: str = "concat"
    dataset: str = "mnist"


class FusionClassifier(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, fusion: str, num_classes: int):
        super().__init__()
        self.encoder = MultiModalEncoder(d_model=d_model, vocab_size=vocab_size, fusion=fusion)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(images, tokens)
        return self.head(emb)


class MultimodalFusionBaseline:
    def __init__(self):
        self.model: FusionClassifier | None = None

    def train(self, config: MultimodalFusionConfig, logger: Logger) -> None:
        dl = get_dataloader(
            name=config.dataset,
            data_root=config.data_root,
            split="train",
            task="classification",
            batch_size=config.effective_batch_size,
            fast_demo=config.fast_demo,
        )
        model = FusionClassifier(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            fusion=config.fusion,
            num_classes=config.num_classes,
        ).to(config.device)
        self.model = model
        opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        for epoch in range(config.effective_epochs):
            total = 0.0
            for images, labels in dl:
                images = images.to(config.device)
                labels = labels.to(config.device)
                tokens = torch.stack([
                    label_to_tokens(int(l), config.text_seq_len, config.vocab_size)
                    for l in labels
                ], dim=0).to(config.device)
                logits = model(images, tokens)
                loss = F.cross_entropy(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            logger.log_metrics(epoch, {"loss": total / max(1, len(dl))})

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))
