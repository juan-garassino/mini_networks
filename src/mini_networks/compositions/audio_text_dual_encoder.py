"""Audio-text dual-encoder contrastive composition."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.models.multimodal.encoders import AudioConvEncoder, TextEncoder, pool_sequence
from mini_networks.compositions.base import ContrastiveCompositionBase


class AudioTextDualEncoderConfig(BaseConfig):
    model_name: str = "audio_text_dual_encoder"
    d_model: int = 64
    vocab_size: int = 256
    text_seq_len: int = 32
    temperature: float = 0.2
    dataset: str = "speech_digits"
    sample_len: int = 4000
    require_downloads: bool = True


class AudioTextDualEncoder(ContrastiveCompositionBase):
    def __init__(self):
        super().__init__()

    def _build_modules(self, config: AudioTextDualEncoderConfig) -> dict[str, torch.nn.Module]:
        return {
            "audio": AudioConvEncoder(d_model=config.d_model).to(config.device),
            "text": TextEncoder(vocab_size=config.vocab_size, d_model=config.d_model).to(config.device),
        }

    def _get_dataloader(self, config: AudioTextDualEncoderConfig):
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

    def _encode_pair(self, modules, primary, tokens, config: AudioTextDualEncoderConfig):
        a_tokens = modules["audio"](primary)
        t_tokens = modules["text"](tokens)
        a = F.normalize(pool_sequence(a_tokens, "mean"), dim=-1)
        t = F.normalize(pool_sequence(t_tokens, "mean"), dim=-1)
        return a, t

    def _infer_embeddings(self, config: AudioTextDualEncoderConfig, inputs: dict[str, Any]) -> torch.Tensor:
        audio = self.modules["audio"]
        text = self.modules["text"]
        audio.eval()
        text.eval()
        waves = self._as_tensor(inputs, "waves").to(config.device)
        labels = self._as_tensor(inputs, "labels", dtype=torch.long)
        tokens = self._prepare_tokens(labels, config).to(config.device)
        with torch.no_grad():
            a_tokens = audio(waves)
            t_tokens = text(tokens)
            a = F.normalize(pool_sequence(a_tokens, "mean"), dim=-1)
            t = F.normalize(pool_sequence(t_tokens, "mean"), dim=-1)
            return (a + t) / 2
