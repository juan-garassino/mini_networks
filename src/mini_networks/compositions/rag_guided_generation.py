"""RAG-guided generation composition: retrieve context, then generate."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.models.rag.model import NanoRAG
from mini_networks.models.transformer.model import TransformerLM
from mini_networks.models.transformer.tokenizer import CharTokenizer


class RAGGuidedGenerationConfig(BaseConfig):
    model_name: str = "rag_guided_generation"
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    d_ff: int = 128
    seq_len: int = 64
    dropout: float = 0.1
    top_k: int = 3
    chunk_size: int = 200
    dataset: str = "text_file"
    text_file: str = ""


class RAGGuidedGeneration:
    def __init__(self):
        self.model: TransformerLM | None = None
        self.tokenizer: CharTokenizer | None = None
        self.rag: NanoRAG | None = None

    def _build_lm(self, config: RAGGuidedGenerationConfig, vocab_size: int) -> TransformerLM:
        return TransformerLM(
            vocab_size=vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            seq_len=config.seq_len,
            dropout=config.dropout,
        ).to(config.device)

    def train(self, config: RAGGuidedGenerationConfig, logger: Logger) -> None:
        dl = get_dataloader(
            name=config.dataset,
            data_root=config.data_root,
            split="train",
            batch_size=config.effective_batch_size,
            fast_demo=config.effective_fast_demo,
            sample_limit=config.dataset_sample_limit,
            file_path=config.text_file,
            seq_len=config.seq_len,
        )
        ds = dl.dataset
        vocab_size = ds.vocab_size
        self.tokenizer = ds.tokenizer

        model = self._build_lm(config, vocab_size)
        self.model = model
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

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

        corpus = getattr(ds, "text", "")
        self.rag = NanoRAG(top_k=config.top_k, chunk_size=config.chunk_size)
        self.rag.add_documents([corpus])

    def generate(self, config: RAGGuidedGenerationConfig, query: str, max_new_tokens: int = 64) -> str:
        if self.model is None or self.rag is None or self.tokenizer is None:
            raise RuntimeError("Train first.")
        context = self.rag.retrieve(query)
        prompt = context + "\n" + query
        ids = self.tokenizer.encode(prompt)
        prompt_ids = torch.tensor([ids], dtype=torch.long, device=config.device)
        out = self.model.generate(prompt_ids, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(out[0].tolist())
