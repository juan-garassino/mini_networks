"""Colab bootstrap: install deps, select model or composition, run training.

Interactive menu
----------------
  In a Colab cell:
      from mini_networks.colab.launcher import interactive_menu
      interactive_menu()

  Or pick a specific model:
      from mini_networks.colab.launcher import run_model, run_composition
      run_model("clip", fast_demo=True)
      run_composition("clip_guided_diffusion", fast_demo=True)

CLI (non-interactive)
---------------------
  uv run python -m mini_networks.colab.launcher --model clip --fast_demo
  uv run python -m mini_networks.colab.launcher --composition gan_diffusion_comparison
  uv run python -m mini_networks.colab.launcher --interactive
"""
from __future__ import annotations

import datetime
import os
import subprocess
import sys
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

console = Console()


# ---------------------------------------------------------------------------
# Available items
# ---------------------------------------------------------------------------

from mini_networks.core.registry import MODEL_NAMES as MODELS

COMPOSITIONS = [
    "clip_guided_diffusion",
    "transformer_clip_diffusion",
    "gan_diffusion_comparison",
    "clip_guided_gan",
    "classifier_guided_diffusion",
    "rag_guided_generation",
    "lora_lm",
    "segment_then_detect",
    "multitask_vision",
    "diffusion_distillation",
    "audio_text_contrastive",
    "tabular_text_cross_attention",
    "audio_text_dual_encoder",
    "tabular_text_dual_encoder",
    "classifier_guided_gan",
    "rag_conditioned_diffusion",
    "image_captioning",
    "multimodal_fusion_baseline",
    "latent_diffusion",
]

_DESCRIPTIONS = {
    "clip":                          "Contrastive image–text matching on MNIST",
    "diffusion":                     "DDPM denoising with EMA + curriculum learning",
    "segmentation":                  "UNet binary / multiclass segmentation on MNIST",
    "detection":                     "YOLO-style digit localisation on 56×56 canvas",
    "classifier":                    "Small CNN classifier baseline on MNIST/Fashion",
    "resnet":                        "Mini ResNet baseline on MNIST/Fashion",
    "vit":                           "Mini ViT baseline on MNIST/Fashion",
    "vae":                           "Conv VAE reconstruction on MNIST/Fashion",
    "unet_ae":                       "UNet autoencoder reconstruction",
    "simclr":                        "SimCLR-lite contrastive vision pretraining",
    "transformer":                   "Character-level TransformerLM on Shakespeare",
    "mamba":                         "NanoMamba state-space sequence model",
    "gan":                           "Generator + Discriminator trained on MNIST",
    "rnn":                           "RNN / LSTM / GRU recurrent language model",
    "lora":                          "Low-rank fine-tuning: MNIST → FashionMNIST",
    "rag":                           "TF-IDF retrieval + TransformerLM generation",
    "rl_maze":                       "Q / DQN / PPO agents on a procedural maze",
    "rlhf":                          "PPO fine-tuning with Shakespearean reward",
    "reinforce":                     "REINFORCE policy gradient on a procedural maze",
    "audio_classifier":              "1D CNN classifier on speech digits",
    "audio_spectrogram":             "2D CNN on audio spectrograms",
    "audio_transformer":             "Transformer over spectrogram frames",
    "audio_melspectrogram":          "2D CNN on mel-spectrograms",
    "tabular_classifier":            "MLP classifier on Iris (tabular)",
    "tabular_diffusion":             "Diffusion for tabular data synthesis",
    "mobilenet":                     "Tiny MobileNet-like CNN baseline",
    "convnext":                      "Tiny ConvNeXt-like CNN baseline",
    "vision_embed":                  "Vision embedding encoder (contrastive)",
    "text_seq2seq":                  "Transformer encoder-decoder (seq2seq)",
    "text_token_classifier":         "Token classifier (vowel vs other)",
    "pixelcnn":                      "PixelCNN-lite autoregressive model",
    "clip_guided_diffusion":         "CLIP + Diffusion — text-guided image generation",
    "transformer_clip_diffusion":    "Transformer + CLIP + Diffusion — LM steers generation",
    "gan_diffusion_comparison":      "GAN vs Diffusion — side-by-side educational comparison",
    "clip_guided_gan":               "GAN guided by CLIP similarity",
    "classifier_guided_diffusion":   "Classifier-guided diffusion sampling",
    "rag_guided_generation":         "Retrieve context then generate (RAG-guided)",
    "lora_lm":                        "LoRA adapter fine-tuning for TransformerLM",
    "segment_then_detect":           "Segmentation then bbox detection",
    "multitask_vision":              "Shared encoder with cls + seg + det heads",
    "diffusion_distillation":        "Distill diffusion teacher into small denoiser",
    "audio_text_contrastive":        "Audio-text contrastive alignment (speech digits)",
    "tabular_text_cross_attention":  "Tabular-text cross-attention alignment (Iris)",
    "audio_text_dual_encoder":       "Audio-text dual-encoder contrastive",
    "tabular_text_dual_encoder":     "Tabular-text dual-encoder contrastive",
    "classifier_guided_gan":         "Classifier-guided GAN",
    "rag_conditioned_diffusion":     "RAG-conditioned diffusion",
    "image_captioning":              "Image captioning (MNIST)",
    "multimodal_fusion_baseline":    "Image+text fusion classifier",
    "latent_diffusion":              "Latent diffusion (VAE + UNet)",
}

_CATEGORY = {name: "Vision / Multimodal" for name in [
    "clip", "diffusion", "segmentation", "detection", "gan",
    "classifier", "resnet", "vit", "vae", "unet_ae", "simclr",
]}
_CATEGORY.update({name: "Language" for name in ["transformer", "mamba", "rnn", "lora", "rag", "rlhf"]})
_CATEGORY["rl_maze"] = "Reinforcement Learning"
_CATEGORY["reinforce"] = "Reinforcement Learning"
_CATEGORY["audio_classifier"] = "Audio"
_CATEGORY["audio_spectrogram"] = "Audio"
_CATEGORY["audio_transformer"] = "Audio"
_CATEGORY["audio_melspectrogram"] = "Audio"
_CATEGORY["tabular_classifier"] = "Tabular"
_CATEGORY["tabular_diffusion"] = "Tabular"
_CATEGORY["mobilenet"] = "Vision / Multimodal"
_CATEGORY["convnext"] = "Vision / Multimodal"
_CATEGORY["vision_embed"] = "Vision / Multimodal"
_CATEGORY["text_seq2seq"] = "Language"
_CATEGORY["text_token_classifier"] = "Language"
_CATEGORY["pixelcnn"] = "Vision / Multimodal"
_CATEGORY["clip_guided_diffusion"] = "Composition"
_CATEGORY["transformer_clip_diffusion"] = "Composition"
_CATEGORY["gan_diffusion_comparison"] = "Composition"
_CATEGORY["clip_guided_gan"] = "Composition"
_CATEGORY["classifier_guided_diffusion"] = "Composition"
_CATEGORY["rag_guided_generation"] = "Composition"
_CATEGORY["lora_lm"] = "Composition"
_CATEGORY["segment_then_detect"] = "Composition"
_CATEGORY["multitask_vision"] = "Composition"
_CATEGORY["diffusion_distillation"] = "Composition"
_CATEGORY["audio_text_contrastive"] = "Composition"
_CATEGORY["tabular_text_cross_attention"] = "Composition"
_CATEGORY["audio_text_dual_encoder"] = "Composition"
_CATEGORY["tabular_text_dual_encoder"] = "Composition"
_CATEGORY["classifier_guided_gan"] = "Composition"
_CATEGORY["rag_conditioned_diffusion"] = "Composition"
_CATEGORY["image_captioning"] = "Composition"
_CATEGORY["multimodal_fusion_baseline"] = "Composition"
_CATEGORY["latent_diffusion"] = "Composition"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def install_deps() -> None:
    """Install the package in the current Python environment (pip-based)."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".[dev]", "-q"])


def _ts() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _run_base(checkpoint_root: str, name: str) -> str:
    return os.path.join(checkpoint_root, name)


def _make_models_table(items: list[str], title: str) -> Table:
    tbl = Table(title=title, box=box.SIMPLE_HEAD, show_lines=False, highlight=True)
    tbl.add_column("#", style="bold cyan", justify="right", width=4)
    tbl.add_column("Name", style="bold white", width=30)
    tbl.add_column("Category", style="dim", width=22)
    tbl.add_column("Description", style="white")
    for i, name in enumerate(items, 1):
        tbl.add_row(str(i), name, _CATEGORY.get(name, ""), _DESCRIPTIONS.get(name, ""))
    return tbl


def _probe_preview(value: Any) -> str:
    import torch

    if isinstance(value, torch.Tensor):
        return f"tensor{tuple(value.shape)}"
    if isinstance(value, str):
        return value[:48] + ("..." if len(value) > 48 else "")
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}[{len(value)}]"
    if isinstance(value, dict):
        return f"dict[{', '.join(value.keys())}]"
    return type(value).__name__


def _validate_probe_output(output: Any) -> str:
    import torch

    if output is None:
        raise RuntimeError("Inference probe returned no output.")
    if isinstance(output, dict):
        payload = {k: v for k, v in output.items() if k not in {"config", "run_dir"}}
        if not payload:
            raise RuntimeError("Inference probe returned only metadata.")
        parts = []
        for key, value in payload.items():
            if isinstance(value, torch.Tensor) and value.numel() == 0:
                raise RuntimeError(f"Inference probe returned empty tensor for {key}.")
            if isinstance(value, (list, tuple)) and len(value) == 0:
                raise RuntimeError(f"Inference probe returned empty collection for {key}.")
            if isinstance(value, str) and not value.strip():
                raise RuntimeError(f"Inference probe returned empty text for {key}.")
            parts.append(f"{key}={_probe_preview(value)}")
        return ", ".join(parts)
    if isinstance(output, torch.Tensor):
        if output.numel() == 0:
            raise RuntimeError("Inference probe returned an empty tensor.")
        return _probe_preview(output)
    if isinstance(output, str):
        if not output.strip():
            raise RuntimeError("Inference probe returned empty text.")
        return _probe_preview(output)
    if isinstance(output, (list, tuple)) and len(output) == 0:
        raise RuntimeError("Inference probe returned an empty collection.")
    return _probe_preview(output)


def _run_model_inference_probe(model: str, trainer: Any, config: Any, dataloader: Any) -> str:
    batch = None
    batch_models = {
        "clip",
        "segmentation",
        "detection",
        "classifier",
        "resnet",
        "vit",
        "vae",
        "unet_ae",
        "simclr",
        "lora",
        "audio_classifier",
        "audio_spectrogram",
        "audio_transformer",
        "audio_melspectrogram",
        "tabular_classifier",
        "mobilenet",
        "convnext",
        "vision_embed",
        "text_seq2seq",
        "text_token_classifier",
    }
    if model in batch_models:
        batch = next(iter(dataloader))

    if model in {"classifier", "resnet", "vit", "mobilenet", "convnext"}:
        output = trainer.infer(config, batch[0][:1])
    elif model == "clip":
        output = trainer.infer(config, {"images": batch[0][:1]})
    elif model == "segmentation":
        output = trainer.infer(config, {"images": batch[0][:1]})
    elif model == "detection":
        output = trainer.infer(config, {"images": batch[0][:1]})
    elif model in {"vae"}:
        output = trainer.infer(config, {"sample": 1})
    elif model in {"unet_ae", "audio_classifier", "audio_spectrogram", "audio_transformer", "audio_melspectrogram"}:
        output = trainer.infer(config, batch[0][:1])
    elif model in {"simclr", "vision_embed"}:
        output = trainer.infer(config, {"images": batch[0][:1]})
    elif model in {"diffusion", "gan", "pixelcnn", "tabular_diffusion"}:
        output = trainer.infer(config, {"n_samples": 1})
    elif model in {"transformer", "mamba", "rnn", "rlhf"}:
        output = trainer.infer(config, {"prompt": "To be", "max_new_tokens": 8})
    elif model == "rag":
        output = trainer.infer(config, {"query": "To be", "max_new_tokens": 8})
    elif model in {"rl_maze", "reinforce"}:
        output = trainer.infer(config, {})
    elif model == "lora":
        output = trainer.infer(config, {"images": batch[0][:1]})
    elif model == "tabular_classifier":
        output = trainer.infer(config, {"features": batch[0][:1]})
    elif model == "text_seq2seq":
        output = trainer.infer(config, {"src": batch[0][0]})
    elif model == "text_token_classifier":
        output = trainer.infer(config, {"tokens": batch[0][0]})
    else:
        raise RuntimeError(f"No inference probe implemented for model {model}.")

    return _validate_probe_output(output)


def list_models() -> None:
    console.print(_make_models_table(MODELS, "Available Models"))


def list_compositions() -> None:
    console.print(_make_models_table(COMPOSITIONS, "Available Compositions"))


# ---------------------------------------------------------------------------
# Single-model runner
# ---------------------------------------------------------------------------

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
        f"[bold]{_DESCRIPTIONS.get(model, model)}[/bold]\n"
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
    if composition not in COMPOSITIONS:
        raise ValueError(f"Unknown composition: {composition!r}. Available: {COMPOSITIONS}")

    console.print(Panel(
        f"[bold]{_DESCRIPTIONS.get(composition, composition)}[/bold]\n"
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
        if composition == "clip_guided_diffusion":
            result = _run_clip_guided_diffusion(fast_demo, training_tier, data_root, device, checkpoint_root)
        elif composition == "transformer_clip_diffusion":
            result = _run_transformer_clip_diffusion(fast_demo, training_tier, data_root, device, checkpoint_root)
        elif composition == "gan_diffusion_comparison":
            result = _run_gan_diffusion_comparison(fast_demo, training_tier, data_root, device, checkpoint_root)
        elif composition == "clip_guided_gan":
            result = _run_clip_guided_gan(fast_demo, training_tier, data_root, device, checkpoint_root)
        elif composition == "classifier_guided_diffusion":
            result = _run_classifier_guided_diffusion(fast_demo, training_tier, data_root, device, checkpoint_root)
        elif composition == "rag_guided_generation":
            result = _run_rag_guided_generation(fast_demo, training_tier, data_root, device, checkpoint_root)
        elif composition == "lora_lm":
            result = _run_lora_lm(fast_demo, training_tier, data_root, device, checkpoint_root)
        elif composition == "segment_then_detect":
            result = _run_segment_then_detect(fast_demo, training_tier, data_root, device, checkpoint_root)
        elif composition == "multitask_vision":
            result = _run_multitask_vision(fast_demo, training_tier, data_root, device, checkpoint_root)
        elif composition == "diffusion_distillation":
            result = _run_diffusion_distillation(fast_demo, training_tier, data_root, device, checkpoint_root)
        elif composition == "audio_text_contrastive":
            result = _run_audio_text_contrastive(fast_demo, training_tier, data_root, device, checkpoint_root)
        elif composition == "tabular_text_cross_attention":
            result = _run_tabular_text_cross_attention(fast_demo, training_tier, data_root, device, checkpoint_root)
        elif composition == "audio_text_dual_encoder":
            result = _run_audio_text_dual_encoder(fast_demo, training_tier, data_root, device, checkpoint_root)
        elif composition == "tabular_text_dual_encoder":
            result = _run_tabular_text_dual_encoder(fast_demo, training_tier, data_root, device, checkpoint_root)
        elif composition == "classifier_guided_gan":
            result = _run_classifier_guided_gan(fast_demo, training_tier, data_root, device, checkpoint_root)
        elif composition == "rag_conditioned_diffusion":
            result = _run_rag_conditioned_diffusion(fast_demo, training_tier, data_root, device, checkpoint_root)
        elif composition == "image_captioning":
            result = _run_image_captioning(fast_demo, training_tier, data_root, device, checkpoint_root)
        elif composition == "multimodal_fusion_baseline":
            result = _run_multimodal_fusion_baseline(fast_demo, training_tier, data_root, device, checkpoint_root)
        elif composition == "latent_diffusion":
            result = _run_latent_diffusion(fast_demo, training_tier, data_root, device, checkpoint_root)
        else:
            raise RuntimeError(f"No runner implemented for {composition!r}")

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


# ---------------------------------------------------------------------------
# Interactive menu (Colab-friendly)
# ---------------------------------------------------------------------------

def interactive_menu() -> None:
    """Display an interactive text menu for choosing a model or composition."""
    console.print(Panel(
        "[bold cyan]mini_networks[/bold cyan] — Educational ML Playground\n"
        f"[dim]{len(MODELS)} models · {len(COMPOSITIONS)} compositions · unified logging · FastAPI[/dim]",
        border_style="bright_blue",
    ))

    console.print("\nWhat would you like to explore?")
    console.print("  [bold cyan][1][/bold cyan] Train a single model")
    console.print("  [bold magenta][2][/bold magenta] Run a multi-model composition")
    console.print("  [bold red][q][/bold red] Quit")

    choice = console.input("\n[bold]Enter choice:[/bold] ").strip().lower()
    if choice == "q":
        return
    if choice not in ("1", "2"):
        console.print("[red]Invalid choice.[/red]")
        return

    if choice == "1":
        console.print()
        console.print(_make_models_table(MODELS, "Available Models"))
        idx = console.input("\n[bold]Enter model number or name:[/bold] ").strip()
        try:
            model = MODELS[int(idx) - 1] if idx.isdigit() else idx
        except IndexError:
            console.print("[red]Invalid selection.[/red]")
            return
        if model not in MODELS:
            console.print(f"[red]Unknown model: {model!r}[/red]")
            return
    else:
        console.print()
        console.print(_make_models_table(COMPOSITIONS, "Available Compositions"))
        idx = console.input("\n[bold]Enter composition number or name:[/bold] ").strip()
        try:
            comp = COMPOSITIONS[int(idx) - 1] if idx.isdigit() else idx
        except IndexError:
            console.print("[red]Invalid selection.[/red]")
            return
        if comp not in COMPOSITIONS:
            console.print(f"[red]Unknown composition: {comp!r}[/red]")
            return

    tier = (console.input("[bold]Training tier[/bold] [S/M/L] (default: M): ").strip().upper() or "M")
    if tier not in {"S", "M", "L"}:
        console.print("[red]Invalid tier.[/red]")
        return
    fast_demo = tier == "S"
    device = console.input("[bold]Device[/bold] [cpu/cuda/mps] (default: cpu): ").strip() or "cpu"

    if choice == "1":
        run_model(model, fast_demo=fast_demo, training_tier=tier, device=device)
    else:
        run_composition(comp, fast_demo=fast_demo, training_tier=tier, device=device)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="mini_networks training launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Models:       " + ", ".join(MODELS) + "\n"
            "Compositions: " + ", ".join(COMPOSITIONS)
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model",       choices=MODELS,       help="Single model to train")
    group.add_argument("--composition", choices=COMPOSITIONS, help="Multi-model composition to run")
    parser.add_argument("--interactive", action="store_true", help="Show interactive menu")
    parser.add_argument("--list",        action="store_true", help="List all models and compositions")
    parser.add_argument("--epochs",    type=int,   default=2)
    parser.add_argument("--fast_demo", action="store_true", default=True)
    parser.add_argument("--no_fast",   action="store_true", help="Disable fast_demo")
    parser.add_argument("--training_tier", choices=["S", "M", "L"], default="M")
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--data_root", default="/tmp/mini_networks_data")
    parser.add_argument("--checkpoint_root", default=os.path.join(os.getcwd(), "runs"))
    parser.add_argument("--no_resume", action="store_true", help="Disable auto-resume for single-model runs")

    args = parser.parse_args()

    if args.list:
        list_models()
        list_compositions()
    elif args.interactive or (not args.model and not args.composition):
        interactive_menu()
    elif args.model:
        run_model(
            args.model,
            epochs=args.epochs,
            fast_demo=not args.no_fast,
            training_tier="S" if not args.no_fast else args.training_tier,
            data_root=args.data_root,
            device=args.device,
            checkpoint_root=args.checkpoint_root,
            resume=not args.no_resume,
        )
    else:
        run_composition(
            args.composition,
            fast_demo=not args.no_fast,
            training_tier="S" if not args.no_fast else args.training_tier,
            data_root=args.data_root,
            device=args.device,
            checkpoint_root=args.checkpoint_root,
        )
