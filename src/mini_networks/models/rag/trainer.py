"""RAG trainer — trains a TransformerLM on the corpus, builds TF-IDF index."""
from __future__ import annotations

import os
import urllib.request
from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import TextFileDataset
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.rag.config import RAGConfig
from mini_networks.models.rag.model import NanoRAG
from mini_networks.models.transformer.model import TransformerLM
from mini_networks.models.transformer.tokenizer import CharTokenizer
from mini_networks.models.transformer.trainer import _get_shakespeare

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class RAGTrainer(BaseTrainer):
    def __init__(self):
        self.model: TransformerLM | None = None
        self.tokenizer: CharTokenizer | None = None
        self.rag: NanoRAG | None = None

    def _build_lm(self, config: RAGConfig, vocab_size: int) -> TransformerLM:
        return TransformerLM(
            vocab_size=vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            seq_len=config.seq_len,
            dropout=config.dropout,
        ).to(config.device)

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, RAGConfig)

        ds = dataloader.dataset
        if hasattr(ds, "tokenizer"):
            self.tokenizer = ds.tokenizer
            vocab_size = ds.vocab_size
        else:
            vocab_size = config.vocab_size

        effective_config = config.model_copy(update={"vocab_size": vocab_size})
        model = self._build_lm(effective_config, vocab_size)
        self.model = model

        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        logger.log_config(effective_config.model_dump())

        for epoch in range(config.effective_epochs):
            model.train()
            total_loss = 0.0
            for x, y in dataloader:
                x, y = x.to(config.device), y.to(config.device)
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
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

        # Build RAG index from the corpus
        text_file = config.text_file or _get_shakespeare(config.data_root)
        with open(text_file, "r", encoding="utf-8") as f:
            corpus = f.read()
        if config.fast_demo:
            corpus = corpus[:8192]

        self.rag = NanoRAG(top_k=config.top_k, chunk_size=config.chunk_size)
        self.rag.add_documents([corpus])

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        assert isinstance(config, RAGConfig)
        ds = dataloader.dataset
        vocab_size = ds.vocab_size if hasattr(ds, "vocab_size") else config.vocab_size
        if self.model is None:
            self.model = self._build_lm(config, vocab_size)
        model = self.model
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(config.device), y.to(config.device)
                logits, _ = model(x)
                total_loss += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
        return {"eval_loss": total_loss / max(1, len(dataloader))}

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, RAGConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if self.rag is None:
            raise RuntimeError("RAG index not built. Call train() first.")
        query = inputs.get("query", "") if isinstance(inputs, dict) else str(inputs)
        max_new = inputs.get("max_new_tokens", 64) if isinstance(inputs, dict) else 64
        temperature = inputs.get("temperature", 1.0) if isinstance(inputs, dict) else 1.0

        generated = self.rag.generate(
            query=query,
            model=self.model,
            tokenizer=self.tokenizer,
            device=config.device,
            max_new_tokens=max_new,
            temperature=temperature,
        )
        context = self.rag.retrieve(query)
        return {"generated": generated, "retrieved": context}


    def load_checkpoint(self, config: BaseConfig, artifacts_dir) -> None:
        """Load model.pt + tokenizer.json. RAG index is not rebuilt (needs corpus)."""
        from pathlib import Path
        assert isinstance(config, RAGConfig)
        path = Path(artifacts_dir)
        state = torch.load(path / "model.pt", map_location=config.device)
        vocab_size = state["token_embed.weight"].shape[0]
        self.model = self._build_lm(config, vocab_size)
        self.model.load_state_dict(state)
        self.model.eval()
        tok_path = path / "tokenizer.json"
        if tok_path.exists():
            self.tokenizer = CharTokenizer.load(str(tok_path))
        # Note: RAG index (self.rag) is not rebuilt from checkpoint.
        # Call train() to rebuild the full pipeline including the TF-IDF index.


def make_rag_dataloader(config: RAGConfig, split: str = "train") -> DataLoader:
    text_file = config.text_file or _get_shakespeare(config.data_root)
    ds = TextFileDataset(
        file_path=text_file,
        seq_len=config.seq_len,
        fast_demo=config.fast_demo,
    )
    return DataLoader(
        ds,
        batch_size=config.effective_batch_size,
        shuffle=(split == "train"),
        num_workers=0,
    )
