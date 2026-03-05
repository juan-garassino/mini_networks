from __future__ import annotations
from mini_networks.core.config import BaseConfig


class LoRAConfig(BaseConfig):
    model_name: str = "lora"

    # CNN backbone
    hidden_dim: int = 128        # FC1 hidden units
    num_classes: int = 10

    # LoRA adapter
    lora_rank: int = 4
    lora_alpha: float = 4.0      # scale = alpha / rank

    # Two-stage training
    pretrain_epochs: int = 3     # train full model on MNIST
    finetune_epochs: int = 2     # train only LoRA adapters on FashionMNIST
    freeze_conv: bool = True     # freeze conv layers during fine-tune
