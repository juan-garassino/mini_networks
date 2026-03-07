"""LoRA fine-tuning for TransformerLM (output head adapter)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.models.transformer.model import TransformerLM
from mini_networks.models.transformer.tokenizer import CharTokenizer


class LoRALMConfig(BaseConfig):
    model_name: str = "lora_lm"
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    d_ff: int = 128
    seq_len: int = 64
    dropout: float = 0.1
    rank: int = 4
    alpha: float = 8.0
    dataset: str = "text_file"
    text_file: str = ""


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.B(self.A(x)) * (self.alpha / self.rank)

    @property
    def lora_params(self):
        return list(self.A.parameters()) + list(self.B.parameters())


class LoRALM:
    def __init__(self):
        self.model: TransformerLM | None = None
        self.tokenizer: CharTokenizer | None = None

    def _build(self, config: LoRALMConfig, vocab_size: int) -> TransformerLM:
        return TransformerLM(
            vocab_size=vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            seq_len=config.seq_len,
            dropout=config.dropout,
        ).to(config.device)

    def train(self, config: LoRALMConfig, logger: Logger) -> None:
        dl = get_dataloader(
            name=config.dataset,
            data_root=config.data_root,
            split="train",
            batch_size=config.effective_batch_size,
            fast_demo=config.fast_demo,
            file_path=config.text_file,
            seq_len=config.seq_len,
        )
        ds = dl.dataset
        vocab_size = ds.vocab_size
        self.tokenizer = ds.tokenizer

        model = self._build(config, vocab_size)
        # Replace lm_head with LoRA adapter
        model.lm_head = LoRALinear(model.lm_head, rank=config.rank, alpha=config.alpha)
        self.model = model

        # Freeze base model weights
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.lm_head.lora_params:
            p.requires_grad_(True)

        optimizer = torch.optim.Adam(model.lm_head.lora_params, lr=config.learning_rate)

        for epoch in range(config.effective_epochs):
            model.train()
            total = 0.0
            for x, y in dl:
                x, y = x.to(config.device), y.to(config.device)
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            logger.log_metrics(epoch, {"loss": total / max(1, len(dl))})

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))
        self.tokenizer.save(str(logger.artifact_path("tokenizer.json")))

    def generate(self, config: LoRALMConfig, prompt: str, max_new_tokens: int = 64) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Train first.")
        ids = self.tokenizer.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=config.device)
        out = self.model.generate(x, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(out[0].tolist())
