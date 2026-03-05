"""CLIP trainer implementing BaseTrainer contract."""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.clip.config import CLIPConfig
from mini_networks.models.clip.data import MNISTImageTextDataset
from mini_networks.models.clip.model import CLIPModel


class CLIPTrainer(BaseTrainer):
    def __init__(self):
        self.model: CLIPModel | None = None

    def _build(self, config: CLIPConfig) -> CLIPModel:
        return CLIPModel(
            embed_dim=config.embed_dim,
            vocab_size=config.vocab_size,
            text_d_model=config.text_d_model,
            text_n_heads=config.text_n_heads,
            text_n_layers=config.text_n_layers,
            text_seq_len=config.text_seq_len,
            temperature=config.temperature,
        ).to(config.device)

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, CLIPConfig)
        model = self._build(config)
        self.model = model
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        logger.log_config(config.model_dump())

        step = 0
        for epoch in range(config.effective_epochs):
            model.train()
            total_loss = 0.0
            for batch in dataloader:
                images, tokens, _ = batch
                images = images.to(config.device)
                tokens = tokens.to(config.device)
                img_emb, txt_emb = model(images, tokens)
                loss = model.contrastive_loss(img_emb, txt_emb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                step += 1
            avg_loss = total_loss / max(1, len(dataloader))
            logger.log_metrics(epoch, {"loss": avg_loss, "epoch": epoch})

        ckpt = logger.artifact_path("model.pt")
        torch.save(model.state_dict(), ckpt)

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        assert isinstance(config, CLIPConfig)
        if self.model is None:
            self.model = self._build(config)
        model = self.model
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                images, tokens, _ = batch
                images = images.to(config.device)
                tokens = tokens.to(config.device)
                img_emb, txt_emb = model(images, tokens)
                loss = model.contrastive_loss(img_emb, txt_emb)
                total_loss += loss.item()
        return {"eval_loss": total_loss / max(1, len(dataloader))}

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, CLIPConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() first or load a checkpoint.")
        self.model.eval()
        with torch.no_grad():
            if isinstance(inputs, dict):
                if "images" in inputs:
                    images = inputs["images"].to(config.device)
                    return {"image_embeds": self.model.encode_image(images).cpu()}
                if "tokens" in inputs:
                    tokens = inputs["tokens"].to(config.device)
                    return {"text_embeds": self.model.encode_text(tokens).cpu()}
        return {}


def make_clip_dataloader(config: CLIPConfig, split: str = "train") -> DataLoader:
    ds = MNISTImageTextDataset(
        data_root=config.data_root,
        train=(split == "train"),
        seq_len=config.text_seq_len,
        vocab_size=config.vocab_size,
        fast_demo=config.fast_demo,
    )
    return DataLoader(
        ds,
        batch_size=config.effective_batch_size,
        shuffle=(split == "train"),
        num_workers=0,
    )
