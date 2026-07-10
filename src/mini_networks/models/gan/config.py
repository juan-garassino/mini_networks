from __future__ import annotations
from mini_networks.core.config import BaseConfig


class GANConfig(BaseConfig):
    model_name: str = "gan"
    latent_dim: int = 100
    image_size: int = 28
    in_channels: int = 1
    disc_dropout: float = 0.3
    lr: float = 0.0002
    # EMA over the generator (0 disables). GAN sample quality is non-monotone
    # in steps (judge 0.139 @ 2k -> 0.049 @ 3.5k, m-triage-4); the saved
    # checkpoint holds the EMA weights so eval sees the smoothed generator.
    g_ema_decay: float = 0.995
    dataset: str = "mnist"
