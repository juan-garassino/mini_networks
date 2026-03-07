"""Audio classifier trainer."""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer, SupervisedTrainer
from mini_networks.models.audio.config import (
    AudioClassifierConfig,
    AudioSpecClassifierConfig,
    AudioTransformerConfig,
    AudioMelSpecClassifierConfig,
)
from mini_networks.models.audio.model import AudioCNN, AudioSpecCNN, AudioTransformer, AudioMelSpecCNN
from mini_networks.core.data.audio import stft_mag, mel_spec, spec_frames


class AudioClassifierTrainer(SupervisedTrainer):
    def __init__(self):
        self.model: AudioCNN | None = None

    def _build(self, config: AudioClassifierConfig) -> AudioCNN:
        return AudioCNN(n_classes=config.n_classes).to(config.device)

    def _forward(self, model, batch, config: AudioClassifierConfig):
        waves, labels = batch
        waves = waves.to(config.device)
        return model(waves), labels

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, AudioClassifierConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        model = self.model
        model.eval()
        with torch.no_grad():
            waves = inputs if not isinstance(inputs, dict) else inputs.get("waves")
            waves = torch.as_tensor(waves, dtype=torch.float32).to(config.device)
            logits = model(waves)
            preds = logits.argmax(dim=-1)
        return {"predictions": preds.cpu().tolist(), "logits": logits.cpu().tolist()}


def make_audio_dataloader(config: AudioClassifierConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        batch_size=config.effective_batch_size,
        fast_demo=config.fast_demo,
        sample_len=config.sample_len,
        require_downloads=config.require_downloads,
    )


class AudioSpecClassifierTrainer(SupervisedTrainer):
    def __init__(self):
        self.model: AudioSpecCNN | None = None

    def _build(self, config: AudioSpecClassifierConfig) -> AudioSpecCNN:
        return AudioSpecCNN(n_classes=config.n_classes).to(config.device)

    def _to_spec(self, waves: torch.Tensor, config: AudioSpecClassifierConfig) -> torch.Tensor:
        # waves: [B, 1, T] -> spec: [B, 1, F, TT]
        mag = stft_mag(waves, n_fft=config.n_fft, hop_length=config.hop_length)
        return mag.unsqueeze(1)

    def _forward(self, model, batch, config: AudioSpecClassifierConfig):
        waves, labels = batch
        waves = waves.to(config.device)
        spec = self._to_spec(waves, config)
        return model(spec), labels

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, AudioSpecClassifierConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        model = self.model
        model.eval()
        with torch.no_grad():
            waves = inputs if not isinstance(inputs, dict) else inputs.get("waves")
            waves = torch.as_tensor(waves, dtype=torch.float32).to(config.device)
            spec = self._to_spec(waves, config)
            logits = model(spec)
            preds = logits.argmax(dim=-1)
        return {"predictions": preds.cpu().tolist(), "logits": logits.cpu().tolist()}


def make_audio_spec_dataloader(config: AudioSpecClassifierConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        batch_size=config.effective_batch_size,
        fast_demo=config.fast_demo,
        sample_len=config.sample_len,
        require_downloads=config.require_downloads,
    )


class AudioTransformerTrainer(SupervisedTrainer):
    def __init__(self):
        self.model: AudioTransformer | None = None

    def _build(self, config: AudioTransformerConfig) -> AudioTransformer:
        input_dim = config.n_fft // 2 + 1
        return AudioTransformer(
            input_dim=input_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            n_classes=config.n_classes,
        ).to(config.device)

    def _to_frames(self, waves: torch.Tensor, config: AudioTransformerConfig) -> torch.Tensor:
        return spec_frames(waves, n_fft=config.n_fft, hop_length=config.hop_length)

    def _forward(self, model, batch, config: AudioTransformerConfig):
        waves, labels = batch
        waves = waves.to(config.device)
        frames = self._to_frames(waves, config)
        return model(frames), labels

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, AudioTransformerConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        model = self.model
        model.eval()
        with torch.no_grad():
            waves = inputs if not isinstance(inputs, dict) else inputs.get("waves")
            waves = torch.as_tensor(waves, dtype=torch.float32).to(config.device)
            frames = self._to_frames(waves, config)
            logits = model(frames)
            preds = logits.argmax(dim=-1)
        return {"predictions": preds.cpu().tolist(), "logits": logits.cpu().tolist()}


def make_audio_transformer_dataloader(config: AudioTransformerConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        batch_size=config.effective_batch_size,
        fast_demo=config.fast_demo,
        sample_len=config.sample_len,
        require_downloads=config.require_downloads,
    )


class AudioMelSpecClassifierTrainer(SupervisedTrainer):
    def __init__(self):
        self.model: AudioMelSpecCNN | None = None

    def _build(self, config: AudioMelSpecClassifierConfig) -> AudioMelSpecCNN:
        return AudioMelSpecCNN(n_classes=config.n_classes).to(config.device)

    def _to_melspec(self, waves: torch.Tensor, config: AudioMelSpecClassifierConfig) -> torch.Tensor:
        return mel_spec(
            waves,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )

    def _forward(self, model, batch, config: AudioMelSpecClassifierConfig):
        waves, labels = batch
        waves = waves.to(config.device)
        mel = self._to_melspec(waves, config)
        return model(mel), labels

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, AudioMelSpecClassifierConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        model = self.model
        model.eval()
        with torch.no_grad():
            waves = inputs if not isinstance(inputs, dict) else inputs.get("waves")
            waves = torch.as_tensor(waves, dtype=torch.float32).to(config.device)
            mel = self._to_melspec(waves, config)
            logits = model(mel)
            preds = logits.argmax(dim=-1)
        return {"predictions": preds.cpu().tolist(), "logits": logits.cpu().tolist()}


def make_audio_melspec_dataloader(config: AudioMelSpecClassifierConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        batch_size=config.effective_batch_size,
        fast_demo=config.fast_demo,
        sample_len=config.sample_len,
        require_downloads=config.require_downloads,
    )
