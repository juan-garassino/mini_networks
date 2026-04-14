"""Token classification trainer (vowel vs other)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.data.text import vowel_labels
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.text_token_classifier.config import TextTokenClassifierConfig
from mini_networks.models.text_token_classifier.model import TokenClassifier


class TokenLabelDataset(Dataset):
    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        # Label: vowel (1) else 0
        labels = vowel_labels(x, getattr(self.base, "itos", {}))
        return x, labels


class TextTokenClassifierTrainer(BaseTrainer):
    def __init__(self):
        self.model: TokenClassifier | None = None

    def _build(self, config: TextTokenClassifierConfig, vocab_size: int) -> TokenClassifier:
        return TokenClassifier(
            vocab_size=vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            seq_len=config.seq_len,
        ).to(config.device)

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, TextTokenClassifierConfig)
        base_ds = dataloader.dataset
        vocab_size = base_ds.vocab_size if hasattr(base_ds, "vocab_size") else config.vocab_size
        model = self._build(config, vocab_size)
        self.model = model
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        logger.log_config(config.model_dump())

        ds = TokenLabelDataset(base_ds)
        dl = DataLoader(ds, batch_size=config.effective_batch_size, shuffle=True, num_workers=0)

        for epoch in range(config.effective_epochs):
            model.train()
            total = 0.0
            for x, labels in dl:
                x, labels = x.to(config.device), labels.to(config.device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            logger.log_metrics(epoch, {"loss": total / max(1, len(dl))})

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        assert isinstance(config, TextTokenClassifierConfig)
        if self.model is None:
            base_ds = dataloader.dataset
            vocab_size = base_ds.vocab_size if hasattr(base_ds, "vocab_size") else config.vocab_size
            self.model = self._build(config, vocab_size)
        model = self.model
        model.eval()
        total = 0.0
        with torch.no_grad():
            for x, labels in DataLoader(TokenLabelDataset(dataloader.dataset), batch_size=4):
                x, labels = x.to(config.device), labels.to(config.device)
                logits = model(x)
                total += F.cross_entropy(logits.view(-1, 2), labels.view(-1)).item()
        return {"eval_loss": total / max(1, len(dataloader))}

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, TextTokenClassifierConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        tokens = inputs.get("tokens") if isinstance(inputs, dict) else inputs
        tokens = torch.as_tensor(tokens, dtype=torch.long).unsqueeze(0).to(config.device)
        with torch.no_grad():
            logits = self.model(tokens)
            preds = logits.argmax(dim=-1)
        return {"predictions": preds.cpu().tolist()}


def make_text_token_classifier_dataloader(config: TextTokenClassifierConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        batch_size=config.effective_batch_size,
        fast_demo=config.effective_fast_demo,
        sample_limit=config.dataset_sample_limit,
        file_path=config.text_file,
        seq_len=config.seq_len,
    )
