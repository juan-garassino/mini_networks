"""Seq2seq trainer."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.data.text import split_seq_halves
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.text_seq2seq.config import TextSeq2SeqConfig
from mini_networks.models.text_seq2seq.model import Seq2SeqTransformer


class Seq2SeqDataset(Dataset):
    """Split each sequence into src/tgt halves."""

    def __init__(self, base_ds: Dataset):
        self.base = base_ds

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        src, tgt = split_seq_halves(x)
        return src, tgt


class TextSeq2SeqTrainer(BaseTrainer):
    def __init__(self):
        self.model: Seq2SeqTransformer | None = None

    def _build(self, config: TextSeq2SeqConfig, vocab_size: int) -> Seq2SeqTransformer:
        return Seq2SeqTransformer(
            vocab_size=vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            seq_len=config.seq_len,
        ).to(config.device)

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, TextSeq2SeqConfig)
        base_ds = dataloader.dataset
        vocab_size = base_ds.vocab_size if hasattr(base_ds, "vocab_size") else config.vocab_size
        model = self._build(config, vocab_size)
        self.model = model
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        logger.log_config(config.model_dump())

        ds = Seq2SeqDataset(base_ds)
        dl = DataLoader(ds, batch_size=config.effective_batch_size, shuffle=True, num_workers=0)

        for epoch in range(config.effective_epochs):
            model.train()
            total = 0.0
            for src, tgt in dl:
                src, tgt = src.to(config.device), tgt.to(config.device)
                # shift target by one for teacher forcing
                logits = model(src, tgt)
                loss = F.cross_entropy(logits.view(-1, vocab_size), tgt.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            logger.log_metrics(epoch, {"loss": total / max(1, len(dl))})

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        assert isinstance(config, TextSeq2SeqConfig)
        if self.model is None:
            base_ds = dataloader.dataset
            vocab_size = base_ds.vocab_size if hasattr(base_ds, "vocab_size") else config.vocab_size
            self.model = self._build(config, vocab_size)
        model = self.model
        model.eval()
        total = 0.0
        with torch.no_grad():
            for src, tgt in DataLoader(Seq2SeqDataset(dataloader.dataset), batch_size=4):
                src, tgt = src.to(config.device), tgt.to(config.device)
                logits = model(src, tgt)
                total += F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1)).item()
        return {"eval_loss": total / max(1, len(dataloader))}

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, TextSeq2SeqConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        src = inputs.get("src") if isinstance(inputs, dict) else inputs
        src = torch.as_tensor(src, dtype=torch.long).unsqueeze(0).to(config.device)
        # greedy decode (copy length)
        tgt = src.clone()
        with torch.no_grad():
            logits = self.model(src, tgt)
            preds = logits.argmax(dim=-1)
        return {"predictions": preds.cpu().tolist()}


def make_text_seq2seq_dataloader(config: TextSeq2SeqConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        batch_size=config.effective_batch_size,
        fast_demo=config.fast_demo,
        file_path=config.text_file,
        seq_len=config.seq_len,
    )
