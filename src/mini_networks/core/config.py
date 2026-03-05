"""Shared base configuration using Pydantic v2."""
from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    model_name: str = "base"
    data_root: str = Field(default_factory=lambda: os.path.join(os.getcwd(), "data"))
    output_dir: str = Field(default_factory=lambda: os.path.join(os.getcwd(), "runs"))
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    device: str = "cpu"
    seed: int = 42
    fast_demo: bool = False

    @property
    def effective_epochs(self) -> int:
        return 1 if self.fast_demo else self.epochs

    @property
    def effective_batch_size(self) -> int:
        return min(self.batch_size, 16) if self.fast_demo else self.batch_size
