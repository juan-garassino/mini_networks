"""CLIP model: CNN image encoder + Transformer text encoder, contrastive (InfoNCE) loss."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """CNN backbone with BatchNorm for stable training on small images."""

    def __init__(self, embed_dim: int = 128, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x))


class TextEncoder(nn.Module):
    """Transformer text encoder with mask pooling over non-padding tokens.

    Padding token id is 0. Instead of relying on the CLS position (which
    holds no special signal for char-level sequences), we mean-pool the
    transformer output over all *valid* (non-padding) positions.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        seq_len: int = 32,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.embed(tokens) + self.pos(positions)

        # Pass padding mask to transformer so attention ignores pad positions
        pad_mask = (tokens == 0)   # [B, T] — True means "ignore this position"
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        # Mean-pool over valid (non-padding) positions
        valid = (~pad_mask).unsqueeze(-1).float()          # [B, T, 1]
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

        return self.proj(pooled)


class CLIPModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        vocab_size: int = 256,
        text_d_model: int = 64,
        text_n_heads: int = 2,
        text_n_layers: int = 2,
        text_seq_len: int = 32,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            d_model=text_d_model,
            n_heads=text_n_heads,
            n_layers=text_n_layers,
            seq_len=text_seq_len,
            embed_dim=embed_dim,
        )
        self.log_temperature = nn.Parameter(torch.tensor(math.log(1 / temperature)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        z = self.image_encoder(images)
        return F.normalize(z, dim=-1)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        z = self.text_encoder(tokens)
        return F.normalize(z, dim=-1)

    def forward(
        self, images: torch.Tensor, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (image_embeds [B, D], text_embeds [B, D])."""
        return self.encode_image(images), self.encode_text(tokens)

    def contrastive_loss(
        self, image_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> torch.Tensor:
        temperature = self.log_temperature.exp()
        logits = torch.matmul(image_embeds, text_embeds.T) * temperature
        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2
