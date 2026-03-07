"""Tabular-text cross-attention composition."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.models.multimodal.blocks import CrossModalEncoder
from mini_networks.compositions.base import ContrastiveCompositionBase


class TabularTextCrossAttentionConfig(BaseConfig):
    model_name: str = "tabular_text_cross_attention"
    d_model: int = 64
    vocab_size: int = 256
    text_seq_len: int = 32
    n_features: int = 4
    dataset: str = "iris"
    require_downloads: bool = True


class TabularTextCrossAttention(ContrastiveCompositionBase):
    def __init__(self):
        super().__init__()

    def _build(self, config: TabularTextCrossAttentionConfig) -> CrossModalEncoder:
        return CrossModalEncoder(
            modality="tabular",
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            fusion="cross_attention",
            n_features=config.n_features,
        ).to(config.device)

    def _build_modules(self, config: TabularTextCrossAttentionConfig) -> dict[str, torch.nn.Module]:
        return {"model": self._build(config)}

    def _get_dataloader(self, config: TabularTextCrossAttentionConfig):
        return get_dataloader(
            name=config.dataset,
            data_root=config.data_root,
            split="train",
            batch_size=config.effective_batch_size,
            fast_demo=config.fast_demo,
            n_features=config.n_features,
            require_downloads=config.require_downloads,
        )

    def _encode_pair(self, modules, primary, tokens, config: TabularTextCrossAttentionConfig):
        model = modules["model"]
        emb = model(primary, tokens)
        return F.normalize(emb, dim=-1)

    def _infer_embeddings(self, config: TabularTextCrossAttentionConfig, inputs: dict[str, Any]) -> torch.Tensor:
        model = self.modules["model"]
        model.eval()
        feats = self._as_tensor(inputs, "features").to(config.device)
        labels = self._as_tensor(inputs, "labels", dtype=torch.long)
        tokens = self._prepare_tokens(labels, config).to(config.device)
        with torch.no_grad():
            emb = model(feats, tokens)
        return emb
