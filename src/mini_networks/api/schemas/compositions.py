"""Composition request/response schemas."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class ComposeTrainRequest(BaseModel):
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    fast_demo: bool = False
    data_root: str = "/tmp/mini_networks_data"
    device: str = "cpu"
    seed: int = 42
    # Composition-specific overrides
    extra: dict[str, Any] = {}


class ComposeTrainResponse(BaseModel):
    job_id: str
    status: str = "started"
    output_dir: str


class ComposeInferRequest(BaseModel):
    inputs: dict[str, Any] = {}
    checkpoint: Optional[str] = None


class ComposeInferResponse(BaseModel):
    composition: str
    outputs: dict[str, Any] | list | str | int | float | None
