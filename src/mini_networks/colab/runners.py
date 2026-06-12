"""Training runners: run_model resolves the registry and trains one model; run_composition dispatches to a per-composition runner via COMPOSITION_RUNNERS."""
from __future__ import annotations

import datetime
import os
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

from mini_networks.colab.catalog import COMPOSITIONS, DESCRIPTIONS, MODELS
from mini_networks.colab.probes import _run_model_inference_probe, _validate_probe_output

console = Console()


def _ts() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _run_base(checkpoint_root: str, name: str) -> str:
    return os.path.join(checkpoint_root, name)



def run_model(
    model: str,
    epochs: int = 2,
    batch_size: int = 32,
    fast_demo: bool = True,
    training_tier: str = "M",
    data_root: str = "/tmp/mini_networks_data",
    device: str = "cpu",
    checkpoint_root: str = "runs",
    resume: bool = True,
    validate_inference: bool = False,
) -> "Logger":  # noqa: F821
    """Train a single model and return the Logger instance."""
    from mini_networks.core.registry import get_model_registry
    from mini_networks.core.checkpoints import find_resumable_run
    from mini_networks.core.logging.logger import Logger

    registry = get_model_registry()
    if model not in registry:
        raise ValueError(f"Unknown model: {model!r}. Available: {MODELS}")

    ConfigClass, TrainerClass, dataloader_fn = registry[model]
    config = ConfigClass(
        epochs=epochs,
        batch_size=batch_size,
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
        checkpoint_root=checkpoint_root,
        resume=resume,
    )
    model_root = _run_base(checkpoint_root, model)
    resumable_run = find_resumable_run(model_root) if resume else None
    if resumable_run is not None:
        config = config.model_copy(update={"run_name": resumable_run.name, "output_dir": str(resumable_run)})
        logger = Logger(output_dir=str(resumable_run), run_name=resumable_run.name)
        output_dir = str(resumable_run)
    else:
        ts = _ts()
        output_dir = os.path.join(model_root, ts)
        config = config.model_copy(update={"run_name": ts, "output_dir": output_dir})
        logger = Logger(output_dir=output_dir, run_name=ts)

    dataloader = dataloader_fn(config, split="train")
    trainer = TrainerClass()

    console.print(Panel(
        f"[bold]{DESCRIPTIONS.get(model, model)}[/bold]\n"
        f"[dim]tier={config.effective_tier}  epochs={config.effective_epochs}  device={device}[/dim]\n"
        f"[dim]checkpoint_root={checkpoint_root}  resume={resume}[/dim]\n"
        f"[dim]output → {output_dir}[/dim]",
        title=f"[bold cyan]Training: {model}[/bold cyan]",
        border_style="cyan",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Training {model}…", total=None)
        trainer.train(config, dataloader, logger)
        progress.update(task, description=f"[green]Done[/green]")

    metrics = logger.read_metrics()
    _print_metrics_summary(metrics)
    if validate_inference:
        summary = _run_model_inference_probe(model, trainer, config, dataloader)
        console.print(f"[green]Inference:[/green] {summary}")
    console.print(f"[green]Artifacts:[/green] {logger.artifacts_dir}")
    return logger


def _print_metrics_summary(metrics: list[dict]) -> None:
    if not metrics:
        return
    tbl = Table(title="Recent Metrics", box=box.SIMPLE, show_header=True)
    tbl.add_column("Step", justify="right", style="dim")
    tbl.add_column("Key", style="bold")
    tbl.add_column("Value", justify="right", style="cyan")
    for m in metrics[-10:]:
        tbl.add_row(
            str(m.get("step", "")),
            str(m.get("key", "")),
            f"{m.get('value', ''):.4f}" if isinstance(m.get("value"), float) else str(m.get("value", "")),
        )
    console.print(tbl)


# ---------------------------------------------------------------------------
# Composition runners
# ---------------------------------------------------------------------------

def run_composition(
    composition: str,
    fast_demo: bool = True,
    training_tier: str = "M",
    data_root: str = "/tmp/mini_networks_data",
    device: str = "cpu",
    checkpoint_root: str = "runs",
    validate_inference: bool = False,
) -> dict:
    """Run a cross-model composition pipeline."""
    runner = COMPOSITION_RUNNERS.get(composition)
    if runner is None:
        raise ValueError(f"Unknown composition: {composition!r}. Available: {COMPOSITIONS}")

    console.print(Panel(
        f"[bold]{DESCRIPTIONS.get(composition, composition)}[/bold]\n"
        f"[dim]tier={'S' if fast_demo else training_tier}  device={device}[/dim]",
        title=f"[bold magenta]Composition: {composition}[/bold magenta]",
        border_style="magenta",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"Running {composition}…", total=None)
        result = runner(fast_demo, training_tier, data_root, device, checkpoint_root)

    if validate_inference:
        summary = _validate_probe_output(result)
        console.print(f"[green]Inference:[/green] {summary}")
    console.print("[green]Composition complete.[/green]")
    return result


def _make_composition_logger(composition: str, checkpoint_root: str):
    from mini_networks.core.logging.logger import Logger

    ts = _ts()
    output_dir = os.path.join(checkpoint_root, composition, ts)
    return Logger(output_dir=output_dir, run_name=ts)


def _run_clip_guided_diffusion(fast_demo, training_tier, data_root, device, checkpoint_root) -> dict:
    from mini_networks.compositions.clip_guided_diffusion import (
        CLIPGuidedDiffusion,
        CLIPGuidedDiffusionConfig,
    )
    cfg = CLIPGuidedDiffusionConfig(
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
    )
    logger = _make_composition_logger("clip_guided_diffusion", checkpoint_root)
    pipeline = CLIPGuidedDiffusion()
    pipeline.train_all(cfg, logger)
    images, class_id = pipeline.text_to_image("digit zero", cfg)
    console.print(f"  Generated images shape: [cyan]{images.shape}[/cyan]")
    return {"images": images, "class_id": class_id, "config": cfg, "run_dir": str(logger.run_dir)}

def _run_transformer_clip_diffusion(fast_demo, training_tier, data_root, device, checkpoint_root) -> dict:
    from mini_networks.compositions.transformer_clip_diffusion import (
        TransformerCLIPDiffusion,
        TransformerCLIPDiffusionConfig,
    )
    cfg = TransformerCLIPDiffusionConfig(
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
    )
    logger = _make_composition_logger("transformer_clip_diffusion", checkpoint_root)
    pipeline = TransformerCLIPDiffusion()
    pipeline.train_all(cfg, logger)
    images, class_id, prompts = pipeline.generate_image("KING", cfg)
    console.print(f"  Best class: [cyan]{class_id}[/cyan]  Generated shape: [cyan]{images.shape}[/cyan]")
    return {"images": images, "class_id": class_id, "prompts": prompts, "run_dir": str(logger.run_dir)}


def _run_gan_diffusion_comparison(fast_demo, training_tier, data_root, device, checkpoint_root) -> dict:
    from mini_networks.compositions.gan_diffusion_comparison import (
        GANDiffusionComparison,
        GANDiffusionConfig,
    )

    cfg = GANDiffusionConfig(
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
    )
    logger = _make_composition_logger("gan_diffusion_comparison", checkpoint_root)
    cmp = GANDiffusionComparison()
    cmp.train_all(cfg, logger)
    results = cmp.compare(cfg, n_samples=4)
    console.print(
        f"  GAN diversity: [cyan]{results['gan_diversity']:.4f}[/cyan]  "
        f"Diffusion diversity: [cyan]{results['diffusion_diversity']:.4f}[/cyan]"
    )
    results["run_dir"] = str(logger.run_dir)
    return results


def _run_clip_guided_gan(fast_demo, training_tier, data_root, device, checkpoint_root) -> dict:
    from mini_networks.compositions.clip_guided_gan import CLIPGuidedGAN, CLIPGuidedGANConfig

    cfg = CLIPGuidedGANConfig(
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
    )
    logger = _make_composition_logger("clip_guided_gan", checkpoint_root)
    pipeline = CLIPGuidedGAN()
    pipeline.train(cfg, logger)
    images = pipeline.sample(cfg, n=4)
    console.print(f"  Generated images shape: [cyan]{images.shape}[/cyan]")
    return {"images": images, "config": cfg, "run_dir": str(logger.run_dir)}


def _run_classifier_guided_diffusion(fast_demo, training_tier, data_root, device, checkpoint_root) -> dict:
    from mini_networks.compositions.classifier_guided_diffusion import (
        ClassifierGuidedDiffusion,
        ClassifierGuidedDiffusionConfig,
    )

    cfg = ClassifierGuidedDiffusionConfig(
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
    )
    logger = _make_composition_logger("classifier_guided_diffusion", checkpoint_root)
    pipeline = ClassifierGuidedDiffusion()
    pipeline.run(cfg, logger)
    images = pipeline.sample(cfg, n=4)
    console.print(f"  Generated images shape: [cyan]{images.shape}[/cyan]")
    return {"images": images, "config": cfg, "run_dir": str(logger.run_dir)}


def _run_rag_guided_generation(fast_demo, training_tier, data_root, device, checkpoint_root) -> dict:
    from mini_networks.compositions.rag_guided_generation import (
        RAGGuidedGeneration,
        RAGGuidedGenerationConfig,
    )

    cfg = RAGGuidedGenerationConfig(
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
    )
    logger = _make_composition_logger("rag_guided_generation", checkpoint_root)
    pipeline = RAGGuidedGeneration()
    pipeline.train(cfg, logger)
    text = pipeline.generate(cfg, "To be or not to be", max_new_tokens=24)
    console.print(f"  Sample: [cyan]{text[:120]}[/cyan]")
    return {"text": text, "config": cfg, "run_dir": str(logger.run_dir)}


def _run_lora_lm(fast_demo, training_tier, data_root, device, checkpoint_root) -> dict:
    from mini_networks.compositions.lora_lm import LoRALM, LoRALMConfig

    cfg = LoRALMConfig(
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
    )
    logger = _make_composition_logger("lora_lm", checkpoint_root)
    pipeline = LoRALM()
    pipeline.train(cfg, logger)
    text = pipeline.generate(cfg, "Hello", max_new_tokens=16)
    console.print(f"  Sample: [cyan]{text[:120]}[/cyan]")
    return {"text": text, "config": cfg, "run_dir": str(logger.run_dir)}


def _run_segment_then_detect(fast_demo, training_tier, data_root, device, checkpoint_root) -> dict:
    from mini_networks.compositions.segment_then_detect import SegmentThenDetect, SegmentThenDetectConfig
    from mini_networks.core.data.registry import get_dataloader

    cfg = SegmentThenDetectConfig(
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
    )
    logger = _make_composition_logger("segment_then_detect", checkpoint_root)
    pipeline = SegmentThenDetect()
    pipeline.train(cfg, logger)
    dl = get_dataloader(
        name=cfg.dataset,
        data_root=cfg.data_root,
        split="train",
        task="classification",
        batch_size=4,
        fast_demo=cfg.effective_fast_demo,
        sample_limit=cfg.dataset_sample_limit,
    )
    images, _ = next(iter(dl))
    bboxes = pipeline.infer_bbox(cfg, images)
    console.print(f"  BBoxes shape: [cyan]{bboxes.shape}[/cyan]")
    return {"bboxes": bboxes, "config": cfg, "run_dir": str(logger.run_dir)}


def _run_multitask_vision(fast_demo, training_tier, data_root, device, checkpoint_root) -> dict:
    from mini_networks.compositions.multitask_vision import (
        MultiTaskDataset,
        MultiTaskVision,
        MultiTaskVisionConfig,
    )
    import torch
    from torch.utils.data import DataLoader

    cfg = MultiTaskVisionConfig(
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
    )
    logger = _make_composition_logger("multitask_vision", checkpoint_root)
    pipeline = MultiTaskVision()
    pipeline.train(cfg, logger)
    probe_ds = MultiTaskDataset(
        data_root=cfg.data_root,
        train=True,
        canvas_size=cfg.canvas_size,
        fast_demo=cfg.effective_fast_demo,
        dataset=cfg.dataset,
        sample_limit=cfg.dataset_sample_limit,
    )
    images, _, _, _ = next(iter(DataLoader(probe_ds, batch_size=1, shuffle=False, num_workers=0)))
    with torch.no_grad():
        logits, seg, bbox = pipeline.model(images.to(cfg.device))
    return {
        "logits": logits.cpu(),
        "segmentation": seg.cpu(),
        "bboxes": bbox.cpu(),
        "config": cfg,
        "run_dir": str(logger.run_dir),
    }


def _run_diffusion_distillation(fast_demo, training_tier, data_root, device, checkpoint_root) -> dict:
    from mini_networks.compositions.diffusion_distillation import (
        DiffusionDistillation,
        DiffusionDistillationConfig,
    )

    cfg = DiffusionDistillationConfig(
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
    )
    logger = _make_composition_logger("diffusion_distillation", checkpoint_root)
    pipeline = DiffusionDistillation()
    pipeline.train(cfg, logger)
    import torch

    xt = torch.randn(1, 1, 28, 28, device=cfg.device)
    t = torch.zeros(1, dtype=torch.long, device=cfg.device)
    with torch.no_grad():
        pred = pipeline.student(xt, t)
    return {"student_pred": pred.cpu(), "config": cfg, "run_dir": str(logger.run_dir)}


def _run_audio_text_contrastive(fast_demo, training_tier, data_root, device, checkpoint_root) -> dict:
    from mini_networks.compositions.audio_text_contrastive import (
        AudioTextContrastive,
        AudioTextContrastiveConfig,
    )

    cfg = AudioTextContrastiveConfig(
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
    )
    logger = _make_composition_logger("audio_text_contrastive", checkpoint_root)
    pipeline = AudioTextContrastive()
    pipeline.train(cfg, logger)
    dl = pipeline._get_dataloader(cfg)
    waves, labels = next(iter(dl))
    output = pipeline.infer(cfg, {"waves": waves[:1], "labels": labels[:1]})
    output.update({"config": cfg, "run_dir": str(logger.run_dir)})
    return output


def _run_tabular_text_cross_attention(fast_demo, training_tier, data_root, device, checkpoint_root) -> dict:
    from mini_networks.compositions.tabular_text_cross_attention import (
        TabularTextCrossAttention,
        TabularTextCrossAttentionConfig,
    )

    cfg = TabularTextCrossAttentionConfig(
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
    )
    logger = _make_composition_logger("tabular_text_cross_attention", checkpoint_root)
    pipeline = TabularTextCrossAttention()
    pipeline.train(cfg, logger)
    dl = pipeline._get_dataloader(cfg)
    features, labels = next(iter(dl))
    output = pipeline.infer(cfg, {"features": features[:1], "labels": labels[:1]})
    output.update({"config": cfg, "run_dir": str(logger.run_dir)})
    return output


def _run_audio_text_dual_encoder(fast_demo, training_tier, data_root, device, checkpoint_root) -> dict:
    from mini_networks.compositions.audio_text_dual_encoder import (
        AudioTextDualEncoder,
        AudioTextDualEncoderConfig,
    )

    cfg = AudioTextDualEncoderConfig(
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
    )
    logger = _make_composition_logger("audio_text_dual_encoder", checkpoint_root)
    pipeline = AudioTextDualEncoder()
    pipeline.train(cfg, logger)
    dl = pipeline._get_dataloader(cfg)
    waves, labels = next(iter(dl))
    output = pipeline.infer(cfg, {"waves": waves[:1], "labels": labels[:1]})
    output.update({"config": cfg, "run_dir": str(logger.run_dir)})
    return output


def _run_tabular_text_dual_encoder(fast_demo, training_tier, data_root, device, checkpoint_root) -> dict:
    from mini_networks.compositions.tabular_text_dual_encoder import (
        TabularTextDualEncoder,
        TabularTextDualEncoderConfig,
    )

    cfg = TabularTextDualEncoderConfig(
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
    )
    logger = _make_composition_logger("tabular_text_dual_encoder", checkpoint_root)
    pipeline = TabularTextDualEncoder()
    pipeline.train(cfg, logger)
    dl = pipeline._get_dataloader(cfg)
    features, labels = next(iter(dl))
    output = pipeline.infer(cfg, {"features": features[:1], "labels": labels[:1]})
    output.update({"config": cfg, "run_dir": str(logger.run_dir)})
    return output


def _run_classifier_guided_gan(fast_demo, training_tier, data_root, device, checkpoint_root) -> dict:
    from mini_networks.compositions.classifier_guided_gan import (
        ClassifierGuidedGAN,
        ClassifierGuidedGANConfig,
    )

    cfg = ClassifierGuidedGANConfig(
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
    )
    logger = _make_composition_logger("classifier_guided_gan", checkpoint_root)
    pipeline = ClassifierGuidedGAN()
    pipeline.train(cfg, logger)
    import torch

    with torch.no_grad():
        z = torch.randn(1, cfg.latent_dim, device=cfg.device)
        images = pipeline.G(z)
        logits = pipeline.C((images + 1.0) / 2.0)
    return {
        "images": images.cpu(),
        "logits": logits.cpu(),
        "config": cfg,
        "run_dir": str(logger.run_dir),
    }


def _run_rag_conditioned_diffusion(fast_demo, training_tier, data_root, device, checkpoint_root) -> dict:
    from mini_networks.compositions.rag_conditioned_diffusion import (
        RAGConditionedDiffusion,
        RAGConditionedDiffusionConfig,
    )

    cfg = RAGConditionedDiffusionConfig(
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
    )
    logger = _make_composition_logger("rag_conditioned_diffusion", checkpoint_root)
    pipeline = RAGConditionedDiffusion()
    pipeline.train(cfg, logger)
    images, prompt = pipeline.sample(cfg)
    console.print(f"  Prompt: [cyan]{prompt[:80]}[/cyan]")
    console.print(f"  Images: [cyan]{images.shape}[/cyan]")
    return {"images": images, "prompt": prompt, "config": cfg, "run_dir": str(logger.run_dir)}


def _run_image_captioning(fast_demo, training_tier, data_root, device, checkpoint_root) -> dict:
    from mini_networks.compositions.image_captioning import ImageCaptioning, ImageCaptioningConfig
    from mini_networks.core.data.registry import get_dataloader
    from mini_networks.models.clip.data import label_to_tokens
    import torch

    cfg = ImageCaptioningConfig(
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
    )
    logger = _make_composition_logger("image_captioning", checkpoint_root)
    pipeline = ImageCaptioning()
    pipeline.train(cfg, logger)
    dl = get_dataloader(
        name=cfg.dataset,
        data_root=cfg.data_root,
        split="train",
        task="classification",
        batch_size=1,
        fast_demo=cfg.effective_fast_demo,
        sample_limit=cfg.dataset_sample_limit,
    )
    images, labels = next(iter(dl))
    tokens = torch.stack(
        [label_to_tokens(int(labels[0]), cfg.text_seq_len, cfg.vocab_size)],
        dim=0,
    ).to(cfg.device)
    with torch.no_grad():
        logits = pipeline.model(images.to(cfg.device), tokens)
    return {
        "caption_logits": logits.cpu(),
        "config": cfg,
        "run_dir": str(logger.run_dir),
    }


def _run_multimodal_fusion_baseline(fast_demo, training_tier, data_root, device, checkpoint_root) -> dict:
    from mini_networks.compositions.multimodal_fusion_baseline import (
        MultimodalFusionBaseline,
        MultimodalFusionConfig,
    )

    cfg = MultimodalFusionConfig(
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
    )
    logger = _make_composition_logger("multimodal_fusion_baseline", checkpoint_root)
    pipeline = MultimodalFusionBaseline()
    pipeline.train(cfg, logger)
    from mini_networks.core.data.registry import get_dataloader
    from mini_networks.models.clip.data import label_to_tokens
    import torch

    dl = get_dataloader(
        name=cfg.dataset,
        data_root=cfg.data_root,
        split="train",
        task="classification",
        batch_size=1,
        fast_demo=cfg.effective_fast_demo,
        sample_limit=cfg.dataset_sample_limit,
    )
    images, labels = next(iter(dl))
    tokens = torch.stack(
        [label_to_tokens(int(labels[0]), cfg.text_seq_len, cfg.vocab_size)],
        dim=0,
    ).to(cfg.device)
    with torch.no_grad():
        logits = pipeline.model(images.to(cfg.device), tokens)
    return {"logits": logits.cpu(), "config": cfg, "run_dir": str(logger.run_dir)}


def _run_latent_diffusion(fast_demo, training_tier, data_root, device, checkpoint_root) -> dict:
    from mini_networks.compositions.latent_diffusion import LatentDiffusion, LatentDiffusionConfig

    cfg = LatentDiffusionConfig(
        fast_demo=fast_demo,
        training_tier=training_tier,
        data_root=data_root,
        device=device,
    )
    logger = _make_composition_logger("latent_diffusion", checkpoint_root)
    pipeline = LatentDiffusion()
    pipeline.train(cfg, logger)
    images = pipeline.sample(cfg, n=4)
    console.print(f"  Images: [cyan]{images.shape}[/cyan]")
    return {"images": images, "config": cfg, "run_dir": str(logger.run_dir)}


# name → runner; COMPOSITIONS (catalog) and this dict are kept in sync by a unit test
COMPOSITION_RUNNERS = {
    "clip_guided_diffusion": _run_clip_guided_diffusion,
    "transformer_clip_diffusion": _run_transformer_clip_diffusion,
    "gan_diffusion_comparison": _run_gan_diffusion_comparison,
    "clip_guided_gan": _run_clip_guided_gan,
    "classifier_guided_diffusion": _run_classifier_guided_diffusion,
    "rag_guided_generation": _run_rag_guided_generation,
    "lora_lm": _run_lora_lm,
    "segment_then_detect": _run_segment_then_detect,
    "multitask_vision": _run_multitask_vision,
    "diffusion_distillation": _run_diffusion_distillation,
    "audio_text_contrastive": _run_audio_text_contrastive,
    "tabular_text_cross_attention": _run_tabular_text_cross_attention,
    "audio_text_dual_encoder": _run_audio_text_dual_encoder,
    "tabular_text_dual_encoder": _run_tabular_text_dual_encoder,
    "classifier_guided_gan": _run_classifier_guided_gan,
    "rag_conditioned_diffusion": _run_rag_conditioned_diffusion,
    "image_captioning": _run_image_captioning,
    "multimodal_fusion_baseline": _run_multimodal_fusion_baseline,
    "latent_diffusion": _run_latent_diffusion,
}
