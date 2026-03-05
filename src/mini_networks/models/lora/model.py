"""LoRA (Low-Rank Adaptation) applied to a small CNN for MNIST/FashionMNIST.

Architecture
-----------
  Conv1: 1→16, 3×3, ReLU, MaxPool
  Conv2: 16→32, 3×3, ReLU, MaxPool
  FC1:   512→hidden_dim   ← LoRA adapter replaces this layer
  FC2:   hidden_dim→10    ← LoRA adapter replaces this layer

LoRA adapter (StandardLoRALinear)
----------------------------------
  forward(x) = W·x + (B·A·x) * (alpha / rank)

  A: [rank, in_features]  — Kaiming-uniform init (learns direction)
  B: [out_features, rank] — zero init (starts as identity adaptation)

During fine-tuning:
  - Conv layers are frozen (freeze_conv=True)
  - W (original weights) are frozen
  - Only A, B matrices are trained → O(rank*(in+out)) params vs O(in*out)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer with a low-rank adapter.

    The adapter adds  Δ = B @ A  scaled by alpha/rank.
    Original weight W is always frozen once set.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 4.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = alpha / rank

        # Base weight — frozen after pre-training
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_param = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_param, -bound, bound)

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self._lora_enabled = True

    def freeze_base(self) -> None:
        """Freeze W; only A, B will update."""
        self.weight.requires_grad_(False)
        if self.bias_param is not None:
            self.bias_param.requires_grad_(False)

    def unfreeze_base(self) -> None:
        self.weight.requires_grad_(True)
        if self.bias_param is not None:
            self.bias_param.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias_param)
        if self._lora_enabled:
            out = out + (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return out


class LoRACNN(nn.Module):
    """Small CNN with LoRA adapters on its two FC layers.

    Same architecture as the reference legacy/006 model:
      Conv1 (1→16) → Conv2 (16→32) → FC1 (512→hidden) → FC2 (hidden→10)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 10,
        rank: int = 4,
        alpha: float = 4.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # After two 2×2 pools on 28×28 → 7×7×32 = 1568.
        # But reference uses 2 pools: 28→14→7, so 7*7*32=1568.
        # We keep fc_in=1568 for compatibility.
        fc_in = 7 * 7 * 32
        self.fc1 = LoRALinear(fc_in, hidden_dim, rank=rank, alpha=alpha)
        self.fc2 = LoRALinear(hidden_dim, num_classes, rank=rank, alpha=alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # [B, 16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 32,  7,  7]
        x = x.view(x.size(0), -1)              # [B, 1568]
        x = F.relu(self.fc1(x))               # [B, hidden]
        return self.fc2(x)                     # [B, num_classes]

    def freeze_for_finetune(self, freeze_conv: bool = True) -> None:
        """Prepare model for LoRA fine-tuning.

        Freezes base weights of FC layers (and optionally conv layers).
        Only LoRA A, B matrices remain trainable.
        """
        if freeze_conv:
            for p in self.conv1.parameters():
                p.requires_grad_(False)
            for p in self.conv2.parameters():
                p.requires_grad_(False)
        self.fc1.freeze_base()
        self.fc2.freeze_base()

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)
        self.fc1.unfreeze_base()
        self.fc2.unfreeze_base()

    def trainable_params(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]
