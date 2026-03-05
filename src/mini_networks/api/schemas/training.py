"""Training request/response schemas."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel


class TrainRequest(BaseModel):
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    fast_demo: bool = False
    data_root: str = "/tmp/mini_networks_data"
    device: str = "cpu"
    seed: int = 42
    # Model-specific overrides
    extra: dict[str, Any] = {}


class TrainResponse(BaseModel):
    job_id: str
    status: str = "started"
    output_dir: str


class JobStatus(BaseModel):
    job_id: str
    model: str
    status: Literal["pending", "running", "done", "failed"]
    epoch: Optional[int] = None
    loss: Optional[float] = None
    error: Optional[str] = None
    output_dir: Optional[str] = None
    metrics_tail: list[dict] = []
