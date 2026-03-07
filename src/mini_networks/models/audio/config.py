from __future__ import annotations
from mini_networks.core.config import BaseConfig


class AudioClassifierConfig(BaseConfig):
    model_name: str = "audio_classifier"
    n_classes: int = 10
    sample_len: int = 4000
    dataset: str = "speech_digits"
    require_downloads: bool = True


class AudioSpecClassifierConfig(BaseConfig):
    model_name: str = "audio_spectrogram"
    n_classes: int = 10
    sample_len: int = 4000
    n_fft: int = 256
    hop_length: int = 128
    dataset: str = "speech_digits"
    require_downloads: bool = True


class AudioTransformerConfig(BaseConfig):
    model_name: str = "audio_transformer"
    n_classes: int = 10
    sample_len: int = 4000
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    n_fft: int = 256
    hop_length: int = 128
    dataset: str = "speech_digits"
    require_downloads: bool = True


class AudioMelSpecClassifierConfig(BaseConfig):
    model_name: str = "audio_melspectrogram"
    n_classes: int = 10
    sample_len: int = 4000
    n_fft: int = 256
    hop_length: int = 128
    n_mels: int = 64
    dataset: str = "speech_digits"
    require_downloads: bool = True
