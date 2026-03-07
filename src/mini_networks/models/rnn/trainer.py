"""RNNLanguageModel trainer — same runtime contract as TransformerTrainer."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.rnn.config import RNNConfig
from mini_networks.models.rnn.model import RNNLanguageModel
from mini_networks.models.transformer.tokenizer import CharTokenizer


class RNNTrainer(BaseTrainer):
    def __init__(self):
        self.model: RNNLanguageModel | None = None
        self.tokenizer: CharTokenizer | None = None

    def _build(self, config: RNNConfig) -> RNNLanguageModel:
        return RNNLanguageModel(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            seq_len=config.seq_len,
            dropout=config.dropout,
            cell_type=config.cell_type,
        ).to(config.device)

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, RNNConfig)

        ds = dataloader.dataset
        if hasattr(ds, "tokenizer"):
            self.tokenizer = ds.tokenizer
            actual_vocab_size = ds.vocab_size
        else:
            actual_vocab_size = config.vocab_size

        effective_config = config.model_copy(update={"vocab_size": actual_vocab_size})
        model = self._build(effective_config)
        self.model = model
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        logger.log_config(effective_config.model_dump())

        for epoch in range(config.effective_epochs):
            model.train()
            total_loss = 0.0
            for x, y in dataloader:
                x, y = x.to(config.device), y.to(config.device)
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, actual_vocab_size), y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / max(1, len(dataloader))
            logger.log_metrics(epoch, {"loss": avg, "epoch": epoch})
            print(f"  epoch {epoch}  loss {avg:.4f}")

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))
        if self.tokenizer:
            self.tokenizer.save(str(logger.artifact_path("tokenizer.json")))

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        assert isinstance(config, RNNConfig)
        ds = dataloader.dataset
        actual_vocab_size = ds.vocab_size if hasattr(ds, "vocab_size") else config.vocab_size
        if self.model is None:
            effective_config = config.model_copy(update={"vocab_size": actual_vocab_size})
            self.model = self._build(effective_config)
        model = self.model
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(config.device), y.to(config.device)
                logits, _ = model(x)
                total_loss += F.cross_entropy(logits.view(-1, actual_vocab_size), y.view(-1)).item()
        return {"eval_loss": total_loss / max(1, len(dataloader))}

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, RNNConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        model = self.model
        prompt_text = inputs.get("prompt", "") if isinstance(inputs, dict) else ""
        max_new = inputs.get("max_new_tokens", 64) if isinstance(inputs, dict) else 64
        temperature = inputs.get("temperature", 1.0) if isinstance(inputs, dict) else 1.0

        if self.tokenizer and prompt_text:
            ids = self.tokenizer.encode(prompt_text)
            prompt = torch.tensor([ids], dtype=torch.long, device=config.device)
        else:
            prompt = torch.zeros(1, 1, dtype=torch.long, device=config.device)

        output = model.generate(prompt, max_new_tokens=max_new, temperature=temperature)
        if self.tokenizer:
            return {"generated": self.tokenizer.decode(output[0].tolist())}
        return {"tokens": output.cpu().tolist()}


    def load_checkpoint(self, config: BaseConfig, artifacts_dir) -> None:
        """Load model.pt + tokenizer.json. Infers vocab_size from state dict."""
        from pathlib import Path
        import json
        assert isinstance(config, RNNConfig)
        path = Path(artifacts_dir)
        state = torch.load(path / "model.pt", map_location=config.device)
        vocab_size = state["token_embed.weight"].shape[0]
        effective_config = config.model_copy(update={"vocab_size": vocab_size})
        self.model = self._build(effective_config)
        self.model.load_state_dict(state)
        self.model.eval()
        tok_path = path / "tokenizer.json"
        if tok_path.exists():
            self.tokenizer = CharTokenizer.load(str(tok_path))


def make_rnn_dataloader(config: RNNConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        batch_size=config.effective_batch_size,
        fast_demo=config.fast_demo,
        file_path=config.text_file,
        seq_len=config.seq_len,
    )
