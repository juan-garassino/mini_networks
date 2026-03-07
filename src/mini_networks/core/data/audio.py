"""Audio preprocessing utilities."""
from __future__ import annotations

import math
import torch


def stft_mag(
    waves: torch.Tensor,
    n_fft: int = 256,
    hop_length: int = 128,
) -> torch.Tensor:
    """Return STFT magnitude: [B, F, T]. waves: [B, 1, T]."""
    waves = waves.squeeze(1)
    spec = torch.stft(
        waves,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
    )
    return spec.abs()


def mel_filterbank(
    n_fft: int,
    n_mels: int,
    sample_rate: int = 16000,
    f_min: float = 0.0,
    f_max: float = 8000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return mel filterbank matrix [M, F]."""
    def hz_to_mel(hz):
        return 2595 * math.log10(1 + hz / 700.0)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    n_freqs = n_fft // 2 + 1
    m_min, m_max = hz_to_mel(f_min), hz_to_mel(f_max)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2, device=device)
    f_pts = mel_to_hz(m_pts)
    bins = torch.floor((n_fft + 1) * f_pts / sample_rate).long()

    fb = torch.zeros(n_mels, n_freqs, device=device)
    for m in range(1, n_mels + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]
        if f_m_minus == f_m or f_m == f_m_plus:
            continue
        fb[m - 1, f_m_minus:f_m] = (
            torch.arange(f_m_minus, f_m, device=device) - f_m_minus
        ) / (f_m - f_m_minus)
        fb[m - 1, f_m:f_m_plus] = (
            f_m_plus - torch.arange(f_m, f_m_plus, device=device)
        ) / (f_m_plus - f_m)
    return fb


def mel_spec(
    waves: torch.Tensor,
    n_fft: int = 256,
    hop_length: int = 128,
    n_mels: int = 64,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Return mel-spectrogram: [B, 1, M, T]."""
    mag = stft_mag(waves, n_fft=n_fft, hop_length=hop_length)
    fb = mel_filterbank(
        n_fft=n_fft,
        n_mels=n_mels,
        sample_rate=sample_rate,
        device=mag.device,
    )
    mel = torch.matmul(fb, mag)  # [B, M, T]
    return mel.unsqueeze(1)


def spec_frames(
    waves: torch.Tensor,
    n_fft: int = 256,
    hop_length: int = 128,
) -> torch.Tensor:
    """Return spectrogram frames for sequence models: [B, T, F]."""
    mag = stft_mag(waves, n_fft=n_fft, hop_length=hop_length)
    return mag.transpose(1, 2)
