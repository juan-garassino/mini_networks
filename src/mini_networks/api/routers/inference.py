"""Inference endpoints: POST /infer/{model}, GET /infer/{model}/info."""
from __future__ import annotations

import torch  # noqa: F401 (used for tensor serialization below)
from fastapi import APIRouter, HTTPException

from mini_networks.api.dependencies import get_model_registry
from mini_networks.api.schemas.inference import InferRequest, InferResponse
from mini_networks.web.model_catalog import build_model_info

router = APIRouter()

# Cached trainers (loaded on first inference request)
_loaded_trainers: dict[str, tuple] = {}


@router.get("/{model_name}/info")
async def model_info(model_name: str):
    registry = get_model_registry()
    if model_name not in registry:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")
    info = build_model_info(model_name)
    return {"model": model_name, "config_schema": info["config_schema"], "defaults": info["defaults"]}


@router.post("/{model_name}", response_model=InferResponse)
async def run_inference(model_name: str, request: InferRequest):
    registry = get_model_registry()
    if model_name not in registry:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    ConfigClass, TrainerClass, _ = registry[model_name]

    # Build config (fast_demo=True for inference, small)
    config = ConfigClass(fast_demo=True)

    # Load or reuse trainer
    cache_key = f"{model_name}:{request.checkpoint or 'new'}"
    if cache_key not in _loaded_trainers:
        trainer = TrainerClass()
        if request.checkpoint:
            try:
                trainer.load_checkpoint(config, request.checkpoint)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to load checkpoint: {e}")
        _loaded_trainers[cache_key] = trainer
    else:
        trainer = _loaded_trainers[cache_key]

    # Build inputs dict
    inputs = dict(request.inputs)
    inputs.setdefault("n_samples", request.n_samples)
    inputs.setdefault("temperature", request.temperature)
    if request.prompt:
        inputs["prompt"] = request.prompt

    try:
        outputs = trainer.infer(config, inputs)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Serialize tensors to lists
    serialized = {}
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            serialized[k] = v.tolist()
        else:
            serialized[k] = v

    return InferResponse(model=model_name, outputs=serialized)
