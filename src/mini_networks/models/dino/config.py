from __future__ import annotations

from mini_networks.core.config import BaseConfig


class DINOConfig(BaseConfig):
    model_name: str = "dino"
    dataset: str = "mnist"
    # ViT backbone (matches MiniViT defaults so vit and dino stay comparable)
    patch_size: int = 4
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    mlp_dim: int = 128
    # DINO head + self-distillation knobs (paper values, scaled down where noted)
    proj_hidden: int = 128   # paper: 2048
    out_dim: int = 64        # prototype count; paper: 65536
    student_temp: float = 0.1
    teacher_temp: float = 0.04
    # 0.99, not the paper's 0.996: at M-tier's ~500 steps, 0.996 leaves the
    # teacher ~13% initial random weights (0.996^500≈0.14) — the student
    # chases a half-random target (flat loss + kNN 0.20, 2026-07-11 audit).
    ema_decay: float = 0.99
    center_momentum: float = 0.9
