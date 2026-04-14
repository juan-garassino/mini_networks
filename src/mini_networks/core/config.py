"""Shared base configuration using Pydantic v2."""
from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    model_name: str = "base"
    data_root: str = Field(default_factory=lambda: os.path.join(os.getcwd(), "data"))
    output_dir: str = Field(default_factory=lambda: os.path.join(os.getcwd(), "runs"))
    checkpoint_root: str = Field(default_factory=lambda: os.path.join(os.getcwd(), "runs"))
    run_name: str | None = None
    resume: bool = True
    training_tier: Literal["S", "M", "L"] = "M"
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    device: str = "cpu"
    seed: int = 42
    fast_demo: bool = False

    @property
    def effective_tier(self) -> Literal["S", "M", "L"]:
        return "S" if self.fast_demo else self.training_tier

    @property
    def effective_fast_demo(self) -> bool:
        return self.effective_tier == "S"

    @property
    def effective_epochs(self) -> int:
        if self.effective_tier == "S":
            return 1
        if self.effective_tier == "M":
            return min(self.epochs, 3)
        return self.epochs

    @property
    def effective_batch_size(self) -> int:
        if self.effective_tier == "S":
            return min(self.batch_size, 16)
        if self.effective_tier == "M":
            return min(self.batch_size, 32)
        return self.batch_size

    @property
    def dataset_sample_limit(self) -> int | None:
        if self.effective_tier == "S":
            return 256
        if self.effective_tier == "M":
            return 2048
        return None

    def tier_epochs(self, full_epochs: int, medium_cap: int = 2) -> int:
        if self.effective_tier == "S":
            return 1
        if self.effective_tier == "M":
            return min(full_epochs, medium_cap)
        return full_epochs
