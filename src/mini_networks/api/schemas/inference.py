"""Inference request/response schemas."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class InferRequest(BaseModel):
    checkpoint: Optional[str] = None
    run_id: Optional[str] = None  # resolve a local run's artifacts/ as the checkpoint
    inputs: dict[str, Any] = {}
    # Optional model-specific params
    n_samples: int = 4
    temperature: float = 1.0
    prompt: str = ""


class InferResponse(BaseModel):
    model: str
    outputs: dict[str, Any]
