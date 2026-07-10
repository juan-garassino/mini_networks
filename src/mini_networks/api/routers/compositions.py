"""Composition endpoints: train and inference for multi-model pipelines."""
from __future__ import annotations

import os
import traceback
from typing import Any, Callable

from fastapi import APIRouter, BackgroundTasks, HTTPException

from mini_networks.api.dependencies import (
    get_job,
    list_jobs,
    make_job_id,
    make_output_dir,
    register_job,
    update_job,
)
from mini_networks.api.schemas.training import JobStatus
from mini_networks.api.schemas.compositions import (
    ComposeTrainRequest,
    ComposeTrainResponse,
    ComposeInferRequest,
    ComposeInferResponse,
)
from mini_networks.core.logging.logger import Logger

from mini_networks.compositions.clip_guided_diffusion import (
    CLIPGuidedDiffusion,
    CLIPGuidedDiffusionConfig,
)
from mini_networks.compositions.transformer_clip_diffusion import (
    TransformerCLIPDiffusion,
    TransformerCLIPDiffusionConfig,
)
from mini_networks.compositions.gan_diffusion_comparison import (
    GANDiffusionComparison,
    GANDiffusionConfig,
)
from mini_networks.compositions.clip_guided_gan import CLIPGuidedGAN, CLIPGuidedGANConfig
from mini_networks.compositions.classifier_guided_diffusion import (
    ClassifierGuidedDiffusion,
    ClassifierGuidedDiffusionConfig,
)
from mini_networks.compositions.rag_guided_generation import (
    RAGGuidedGeneration,
    RAGGuidedGenerationConfig,
)
from mini_networks.compositions.lora_lm import LoRALM, LoRALMConfig
from mini_networks.compositions.segment_then_detect import SegmentThenDetect, SegmentThenDetectConfig
from mini_networks.compositions.multitask_vision import MultiTaskVision, MultiTaskVisionConfig
from mini_networks.compositions.diffusion_distillation import (
    DiffusionDistillation,
    DiffusionDistillationConfig,
)
from mini_networks.core.data.registry import get_dataloader
from mini_networks.compositions.audio_text_contrastive import (
    AudioTextContrastive,
    AudioTextContrastiveConfig,
)
from mini_networks.compositions.tabular_text_cross_attention import (
    TabularTextCrossAttention,
    TabularTextCrossAttentionConfig,
)
from mini_networks.compositions.audio_text_dual_encoder import (
    AudioTextDualEncoder,
    AudioTextDualEncoderConfig,
)
from mini_networks.compositions.tabular_text_dual_encoder import (
    TabularTextDualEncoder,
    TabularTextDualEncoderConfig,
)
from mini_networks.compositions.classifier_guided_gan import (
    ClassifierGuidedGAN,
    ClassifierGuidedGANConfig,
)
from mini_networks.compositions.rag_conditioned_diffusion import (
    RAGConditionedDiffusion,
    RAGConditionedDiffusionConfig,
)
from mini_networks.compositions.image_captioning import (
    ImageCaptioning,
    ImageCaptioningConfig,
)
from mini_networks.compositions.multimodal_fusion_baseline import (
    MultimodalFusionBaseline,
    MultimodalFusionConfig,
)
from mini_networks.compositions.latent_diffusion import (
    LatentDiffusion,
    LatentDiffusionConfig,
)

router = APIRouter()
RUNS_BASE = os.environ.get("MINI_NETWORKS_RUNS", "runs")


CompositionSpec = dict[str, Any]


def _composition_registry() -> dict[str, CompositionSpec]:
    return {
        "clip_guided_diffusion": {
            "Config": CLIPGuidedDiffusionConfig,
            "Builder": CLIPGuidedDiffusion,
            "train": lambda pipeline, cfg, logger: pipeline.train(cfg, logger),
            "infer": lambda pipeline, cfg, inputs: pipeline.text_to_image(
                inputs.get("prompt", "digit zero"), cfg
            ),
        },
        "transformer_clip_diffusion": {
            "Config": TransformerCLIPDiffusionConfig,
            "Builder": TransformerCLIPDiffusion,
            "train": lambda pipeline, cfg, logger: pipeline.train(cfg, logger),
            "infer": lambda pipeline, cfg, inputs: pipeline.generate_image(
                inputs.get("prompt", "KING"), cfg
            ),
        },
        "gan_diffusion_comparison": {
            "Config": GANDiffusionConfig,
            "Builder": GANDiffusionComparison,
            "train": lambda pipeline, cfg, logger: pipeline.train(cfg, logger),
            "infer": lambda pipeline, cfg, inputs: pipeline.compare(
                cfg, n_samples=int(inputs.get("n_samples", 4))
            ),
        },
        "clip_guided_gan": {
            "Config": CLIPGuidedGANConfig,
            "Builder": CLIPGuidedGAN,
            "train": lambda pipeline, cfg, logger: pipeline.train(cfg, logger),
            "infer": lambda pipeline, cfg, inputs: pipeline.sample(
                cfg, n=int(inputs.get("n_samples", 4))
            ),
        },
        "classifier_guided_diffusion": {
            "Config": ClassifierGuidedDiffusionConfig,
            "Builder": ClassifierGuidedDiffusion,
            "train": lambda pipeline, cfg, logger: pipeline.run(cfg, logger),
            "infer": lambda pipeline, cfg, inputs: pipeline.sample(
                cfg, n=int(inputs.get("n_samples", 4))
            ),
        },
        "rag_guided_generation": {
            "Config": RAGGuidedGenerationConfig,
            "Builder": RAGGuidedGeneration,
            "train": lambda pipeline, cfg, logger: pipeline.train(cfg, logger),
            "infer": lambda pipeline, cfg, inputs: pipeline.generate(
                cfg,
                inputs.get("prompt", "To be or not to be"),
                max_new_tokens=int(inputs.get("max_new_tokens", 64)),
            ),
        },
        "lora_lm": {
            "Config": LoRALMConfig,
            "Builder": LoRALM,
            "train": lambda pipeline, cfg, logger: pipeline.train(cfg, logger),
            "infer": lambda pipeline, cfg, inputs: pipeline.generate(
                cfg,
                inputs.get("prompt", "Hello"),
                max_new_tokens=int(inputs.get("max_new_tokens", 64)),
            ),
        },
        "segment_then_detect": {
            "Config": SegmentThenDetectConfig,
            "Builder": SegmentThenDetect,
            "train": lambda pipeline, cfg, logger: pipeline.train(cfg, logger),
            "infer": lambda pipeline, cfg, inputs: pipeline.infer_bbox(
                cfg, _resolve_images(cfg, inputs)
            ),
        },
        "multitask_vision": {
            "Config": MultiTaskVisionConfig,
            "Builder": MultiTaskVision,
            "train": lambda pipeline, cfg, logger: pipeline.train(cfg, logger),
            "infer": lambda pipeline, cfg, inputs: {"status": "trained"},
        },
        "diffusion_distillation": {
            "Config": DiffusionDistillationConfig,
            "Builder": DiffusionDistillation,
            "train": lambda pipeline, cfg, logger: pipeline.train(cfg, logger),
            "infer": lambda pipeline, cfg, inputs: {"status": "trained"},
        },
        "audio_text_contrastive": {
            "Config": AudioTextContrastiveConfig,
            "Builder": AudioTextContrastive,
            "train": lambda pipeline, cfg, logger: pipeline.train(cfg, logger),
            "infer": lambda pipeline, cfg, inputs: pipeline.infer(cfg, inputs),
        },
        "tabular_text_cross_attention": {
            "Config": TabularTextCrossAttentionConfig,
            "Builder": TabularTextCrossAttention,
            "train": lambda pipeline, cfg, logger: pipeline.train(cfg, logger),
            "infer": lambda pipeline, cfg, inputs: pipeline.infer(cfg, inputs),
        },
        "audio_text_dual_encoder": {
            "Config": AudioTextDualEncoderConfig,
            "Builder": AudioTextDualEncoder,
            "train": lambda pipeline, cfg, logger: pipeline.train(cfg, logger),
            "infer": lambda pipeline, cfg, inputs: pipeline.infer(cfg, inputs),
        },
        "tabular_text_dual_encoder": {
            "Config": TabularTextDualEncoderConfig,
            "Builder": TabularTextDualEncoder,
            "train": lambda pipeline, cfg, logger: pipeline.train(cfg, logger),
            "infer": lambda pipeline, cfg, inputs: pipeline.infer(cfg, inputs),
        },
        "classifier_guided_gan": {
            "Config": ClassifierGuidedGANConfig,
            "Builder": ClassifierGuidedGAN,
            "train": lambda pipeline, cfg, logger: pipeline.train(cfg, logger),
            "infer": lambda pipeline, cfg, inputs: {"status": "trained"},
        },
        "rag_conditioned_diffusion": {
            "Config": RAGConditionedDiffusionConfig,
            "Builder": RAGConditionedDiffusion,
            "train": lambda pipeline, cfg, logger: pipeline.train(cfg, logger),
            "infer": lambda pipeline, cfg, inputs: {"status": "trained"},
        },
        "image_captioning": {
            "Config": ImageCaptioningConfig,
            "Builder": ImageCaptioning,
            "train": lambda pipeline, cfg, logger: pipeline.train(cfg, logger),
            "infer": lambda pipeline, cfg, inputs: {"status": "trained"},
        },
        "multimodal_fusion_baseline": {
            "Config": MultimodalFusionConfig,
            "Builder": MultimodalFusionBaseline,
            "train": lambda pipeline, cfg, logger: pipeline.train(cfg, logger),
            "infer": lambda pipeline, cfg, inputs: {"status": "trained"},
        },
        "latent_diffusion": {
            "Config": LatentDiffusionConfig,
            "Builder": LatentDiffusion,
            "train": lambda pipeline, cfg, logger: pipeline.train(cfg, logger),
            "infer": lambda pipeline, cfg, inputs: {"status": "trained"},
        },
    }


# Cached pipelines (loaded on first inference request)
_loaded_pipelines: dict[str, Any] = {}


def _run_composition_training(job_id: str, name: str, config, output_dir: str):
    try:
        update_job(job_id, status="running")
        logger = Logger(output_dir=output_dir, run_name=job_id)
        spec = _composition_registry()[name]
        pipeline = spec["Builder"]()
        spec["train"](pipeline, config, logger)
        metrics = logger.latest_metrics(5)
        last = metrics[-1] if metrics else {}
        update_job(
            job_id,
            status="done",
            epoch=last.get("value") if last.get("key") == "epoch" else None,
            loss=last.get("value") if last.get("key") == "loss" else None,
            metrics_tail=metrics,
        )
    except Exception:
        update_job(job_id, status="failed", error=traceback.format_exc())


def _resolve_images(cfg, inputs):
    images = inputs.get("images") if isinstance(inputs, dict) else None
    if images is not None:
        import torch
        return torch.as_tensor(images, dtype=torch.float32)
    dl = get_dataloader(
        name=getattr(cfg, "dataset", "mnist"),
        data_root=cfg.data_root,
        split="train",
        task="classification",
        batch_size=4,
        fast_demo=True,
    )
    images, _ = next(iter(dl))
    return images


@router.get("/", response_model=list[JobStatus])
async def list_runs():
    return list_jobs()


@router.post("/{composition_name}", response_model=ComposeTrainResponse)
async def start_composition_training(
    composition_name: str,
    request: ComposeTrainRequest,
    background_tasks: BackgroundTasks,
):
    # Same guard as /train — public showcases must not start trainings.
    if os.environ.get("MN_DISABLE_TRAIN") == "1":
        raise HTTPException(status_code=403, detail="training is disabled on this deployment")
    registry = _composition_registry()
    if composition_name not in registry:
        raise HTTPException(status_code=404, detail=f"Unknown composition: {composition_name}")

    ConfigClass = registry[composition_name]["Config"]
    job_id = make_job_id(composition_name)
    output_dir = make_output_dir(RUNS_BASE, composition_name, job_id)

    config_data = {
        "epochs": request.epochs,
        "batch_size": request.batch_size,
        "learning_rate": request.learning_rate,
        "fast_demo": request.fast_demo,
        "data_root": request.data_root,
        "device": request.device,
        "seed": request.seed,
        "output_dir": output_dir,
    }
    extra = request.extra
    config = ConfigClass(**{**config_data, **extra})

    register_job(job_id, composition_name, output_dir)
    background_tasks.add_task(_run_composition_training, job_id, composition_name, config, output_dir)
    return ComposeTrainResponse(job_id=job_id, status="started", output_dir=output_dir)


@router.post("/{composition_name}/infer", response_model=ComposeInferResponse)
async def run_composition_infer(composition_name: str, request: ComposeInferRequest):
    registry = _composition_registry()
    if composition_name not in registry:
        raise HTTPException(status_code=404, detail=f"Unknown composition: {composition_name}")

    ConfigClass = registry[composition_name]["Config"]
    cfg = ConfigClass(fast_demo=True)

    cache_key = f"{composition_name}:{request.checkpoint or 'new'}"
    if cache_key not in _loaded_pipelines:
        pipeline = registry[composition_name]["Builder"]()
        _loaded_pipelines[cache_key] = pipeline
    else:
        pipeline = _loaded_pipelines[cache_key]

    inputs = request.inputs
    try:
        outputs = registry[composition_name]["infer"](pipeline, cfg, inputs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Serialize tensors to lists
    import torch
    serialized = {}
    if isinstance(outputs, dict):
        for k, v in outputs.items():
            serialized[k] = v.tolist() if isinstance(v, torch.Tensor) else v
        return ComposeInferResponse(composition=composition_name, outputs=serialized)
    if isinstance(outputs, torch.Tensor):
        return ComposeInferResponse(composition=composition_name, outputs=outputs.tolist())
    return ComposeInferResponse(composition=composition_name, outputs=outputs)
