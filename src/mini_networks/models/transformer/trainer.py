"""TransformerLM trainer."""
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
from mini_networks.models.transformer.config import TransformerConfig
from mini_networks.models.transformer.model import TransformerLM
from mini_networks.models.transformer.tokenizer import CharTokenizer


class TransformerTrainer(BaseTrainer):
    def __init__(self):
        self.model: TransformerLM | None = None
        self.tokenizer: CharTokenizer | None = None

    def _build(self, config: TransformerConfig) -> TransformerLM:
        ffn_kwargs: dict = {}
        if config.block_type == "moe":
            ffn_kwargs = dict(
                num_experts=config.moe_num_experts,
                k=config.moe_top_k,
                router_hidden=config.moe_router_hidden,
                balance_loss_weight=config.moe_balance_loss_weight,
                entropy_bonus_weight=config.moe_entropy_bonus,
                shared_scale=config.moe_shared_scale,
                add_gumbel=config.moe_add_gumbel,
                temperature=config.moe_router_temp,
            )
        elif config.block_type == "mamba":
            ffn_kwargs = dict(
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
            )
        return TransformerLM(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            seq_len=config.seq_len,
            dropout=config.dropout,
            block_type=config.block_type,
            **ffn_kwargs,
        ).to(config.device)

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, TransformerConfig)

        # Pick up tokenizer + actual vocab_size from the dataset if available
        ds = dataloader.dataset
        if hasattr(ds, "tokenizer"):
            self.tokenizer = ds.tokenizer
            actual_vocab_size = ds.vocab_size
        else:
            actual_vocab_size = config.vocab_size

        # Rebuild config with correct vocab_size (avoids index-out-of-range on real text)
        effective_config = config.model_copy(update={"vocab_size": actual_vocab_size})
        model = self._build(effective_config)
        self.model = model
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        logger.log_config(effective_config.model_dump())

        for epoch in range(config.effective_epochs):
            model.train()
            total_loss = 0.0
            for x, y in dataloader:
                x = x.to(config.device)
                y = y.to(config.device)
                logits, aux = model(x)
                loss = F.cross_entropy(logits.view(-1, actual_vocab_size), y.view(-1)) + aux
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
        assert isinstance(config, TransformerConfig)
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
                x = x.to(config.device)
                y = y.to(config.device)
                logits, _ = model(x)
                total_loss += F.cross_entropy(logits.view(-1, actual_vocab_size), y.view(-1)).item()
        return {"eval_loss": total_loss / max(1, len(dataloader))}

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, TransformerConfig)
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
            text = self.tokenizer.decode(output[0].tolist())
            return {"generated": text}
        return {"tokens": output.cpu().tolist()}


    def load_checkpoint(self, config: BaseConfig, artifacts_dir) -> None:
        """Load model.pt + tokenizer.json. Infers vocab_size from state dict."""
        from pathlib import Path
        import json
        assert isinstance(config, TransformerConfig)
        path = Path(artifacts_dir)
        state = torch.load(path / "model.pt", map_location=config.device)
        vocab_size = state["token_embed.weight"].shape[0]
        effective_config = config.model_copy(update={"vocab_size": vocab_size})
        self.model = self._build(effective_config)
        self.model.load_state_dict(state)
        self.model.eval()
        tok_path = path / "tokenizer.json"
        if tok_path.exists():
            with open(tok_path) as f:
                data = json.load(f)
            if "merges" in data:
                from mini_networks.models.transformer.tokenizer import BPETokenizer
                self.tokenizer = BPETokenizer.load(str(tok_path))
            else:
                self.tokenizer = CharTokenizer.load(str(tok_path))


def make_transformer_dataloader(
    config: TransformerConfig, split: str = "train"
) -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        batch_size=config.effective_batch_size,
        fast_demo=config.effective_fast_demo,
        sample_limit=config.dataset_sample_limit,
        file_path=config.text_file,
        seq_len=config.seq_len,
        tokenizer_type=config.tokenizer_type,
        bpe_vocab_size=config.bpe_vocab_size,
    )
