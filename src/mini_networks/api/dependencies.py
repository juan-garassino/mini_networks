"""Shared dependencies: job store and model registry."""
from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from mini_networks.api.schemas.training import JobStatus


# In-memory job store
_jobs: dict[str, JobStatus] = {}
_lock = threading.Lock()


def register_job(job_id: str, model: str, output_dir: str) -> None:
    with _lock:
        _jobs[job_id] = JobStatus(
            job_id=job_id,
            model=model,
            status="pending",
            output_dir=output_dir,
        )


def update_job(job_id: str, **kwargs) -> None:
    with _lock:
        if job_id in _jobs:
            job = _jobs[job_id]
            for k, v in kwargs.items():
                setattr(job, k, v)


def get_job(job_id: str) -> JobStatus | None:
    return _jobs.get(job_id)


def list_jobs() -> list[JobStatus]:
    return list(_jobs.values())


def make_job_id(model: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{model}-{ts}"


def make_output_dir(base: str, model: str, job_id: str) -> str:
    path = Path(base) / model / job_id
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


# Model registry: maps model_name → (ConfigClass, TrainerClass, dataloader_fn)
def get_model_registry():
    from mini_networks.models.clip.config import CLIPConfig
    from mini_networks.models.clip.trainer import CLIPTrainer, make_clip_dataloader
    from mini_networks.models.diffusion.config import DiffusionConfig
    from mini_networks.models.diffusion.trainer import DDPMTrainer, make_diffusion_dataloader
    from mini_networks.models.segmentation.config import SegmentationConfig
    from mini_networks.models.segmentation.trainer import SegmentationTrainer, make_segmentation_dataloader
    from mini_networks.models.detection.config import DetectionConfig
    from mini_networks.models.detection.trainer import DetectionTrainer, make_detection_dataloader
    from mini_networks.models.transformer.config import TransformerConfig
    from mini_networks.models.transformer.trainer import TransformerTrainer, make_transformer_dataloader
    from mini_networks.models.mamba.config import MambaConfig
    from mini_networks.models.mamba.trainer import MambaTrainer, make_mamba_dataloader
    from mini_networks.models.gan.config import GANConfig
    from mini_networks.models.gan.trainer import GANTrainer, make_gan_dataloader
    from mini_networks.models.rnn.config import RNNConfig
    from mini_networks.models.rnn.trainer import RNNTrainer, make_rnn_dataloader
    from mini_networks.models.lora.config import LoRAConfig
    from mini_networks.models.lora.trainer import LoRATrainer, make_lora_dataloader
    from mini_networks.models.rag.config import RAGConfig
    from mini_networks.models.rag.trainer import RAGTrainer, make_rag_dataloader
    from mini_networks.models.rl_maze.config import RLMazeConfig
    from mini_networks.models.rl_maze.trainer import RLMazeTrainer, make_rl_maze_dataloader
    from mini_networks.models.rlhf.config import RLHFConfig
    from mini_networks.models.rlhf.trainer import RLHFTrainer, make_rlhf_dataloader

    return {
        "clip": (CLIPConfig, CLIPTrainer, make_clip_dataloader),
        "diffusion": (DiffusionConfig, DDPMTrainer, make_diffusion_dataloader),
        "segmentation": (SegmentationConfig, SegmentationTrainer, make_segmentation_dataloader),
        "detection": (DetectionConfig, DetectionTrainer, make_detection_dataloader),
        "transformer": (TransformerConfig, TransformerTrainer, make_transformer_dataloader),
        "mamba": (MambaConfig, MambaTrainer, make_mamba_dataloader),
        "gan": (GANConfig, GANTrainer, make_gan_dataloader),
        "rnn": (RNNConfig, RNNTrainer, make_rnn_dataloader),
        "lora": (LoRAConfig, LoRATrainer, make_lora_dataloader),
        "rag": (RAGConfig, RAGTrainer, make_rag_dataloader),
        "rl_maze": (RLMazeConfig, RLMazeTrainer, make_rl_maze_dataloader),
        "rlhf": (RLHFConfig, RLHFTrainer, make_rlhf_dataloader),
    }
