from __future__ import annotations
from mini_networks.core.config import BaseConfig


class VAEConfig(BaseConfig):
    model_name: str = "vae"
    latent_dim: int = 32
    hidden_dim: int = 128
    # 0.0015 ≈ 1/784, NOT 1.0: recon is per-PIXEL mean (~0.05) while KL is
    # per-DIM mean (~O(1)), so beta=1 let KL dominate ~20x — the encoder
    # collapsed to the prior and every input reconstructed to the same mean
    # blob (m-vision-7 recon_pairs). Balancing beta to the unit mismatch
    # restores per-sample information; raise it deliberately for the
    # beta-VAE blur-vs-regularity lesson.
    beta: float = 0.0015
    dataset: str = "mnist"
