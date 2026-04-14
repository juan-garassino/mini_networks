"""Audio-text contrastive composition using cross-attention embeddings."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.models.multimodal.blocks import CrossModalEncoder
from mini_networks.compositions.base import ContrastiveCompositionBase


class AudioTextContrastiveConfig(BaseConfig):
    model_name: str = "audio_text_contrastive"
    d_model: int = 64
    vocab_size: int = 256
    text_seq_len: int = 32
    temperature: float = 0.2
    dataset: str = "speech_digits"
    sample_len: int = 4000
    require_downloads: bool = True


class AudioTextContrastive(ContrastiveCompositionBase):
    def __init__(self):
        super().__init__()

    def _build(self, config: AudioTextContrastiveConfig) -> CrossModalEncoder:
        return CrossModalEncoder(
            modality="audio",
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            fusion="cross_attention",
        ).to(config.device)

    def _build_modules(self, config: AudioTextContrastiveConfig) -> dict[str, torch.nn.Module]:
        return {"model": self._build(config)}

    def _get_dataloader(self, config: AudioTextContrastiveConfig):
        return get_dataloader(
            name=config.dataset,
            data_root=config.data_root,
            split="train",
            batch_size=config.effective_batch_size,
            fast_demo=config.effective_fast_demo,
            sample_limit=config.dataset_sample_limit,
            sample_len=config.sample_len,
            require_downloads=config.require_downloads,
        )

    def _encode_pair(self, modules, primary, tokens, config: AudioTextContrastiveConfig):
        model = modules["model"]
        emb = model(primary, tokens)
        return F.normalize(emb, dim=-1)

    def _infer_embeddings(self, config: AudioTextContrastiveConfig, inputs: dict[str, Any]) -> torch.Tensor:
        model = self.modules["model"]
        model.eval()
        waves = self._as_tensor(inputs, "waves").to(config.device)
        labels = self._as_tensor(inputs, "labels", dtype=torch.long)
        tokens = self._prepare_tokens(labels, config).to(config.device)
        with torch.no_grad():
            emb = model(waves, tokens)
        return emb
