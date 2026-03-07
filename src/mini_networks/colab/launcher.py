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
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

console = Console()


# ---------------------------------------------------------------------------
# Available items
# ---------------------------------------------------------------------------

MODELS = [
    "clip",
    "diffusion",
    "segmentation",
    "detection",
    "classifier",
    "resnet",
    "vit",
    "vae",
    "unet_ae",
    "simclr",
    "transformer",
    "mamba",
    "gan",
    "rnn",
    "lora",
    "rag",
    "rl_maze",
    "rlhf",
    "reinforce",
    "audio_classifier",
    "audio_spectrogram",
    "audio_transformer",
    "audio_melspectrogram",
    "tabular_classifier",
    "tabular_diffusion",
    "mobilenet",
    "convnext",
    "vision_embed",
    "text_seq2seq",
    "text_token_classifier",
    "pixelcnn",
]

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


def _make_models_table(items: list[str], title: str) -> Table:
    tbl = Table(title=title, box=box.SIMPLE_HEAD, show_lines=False, highlight=True)
    tbl.add_column("#", style="bold cyan", justify="right", width=4)
    tbl.add_column("Name", style="bold white", width=30)
    tbl.add_column("Category", style="dim", width=22)
    tbl.add_column("Description", style="white")
    for i, name in enumerate(items, 1):
        tbl.add_row(str(i), name, _CATEGORY.get(name, ""), _DESCRIPTIONS.get(name, ""))
    return tbl


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
    data_root: str = "/tmp/mini_networks_data",
    device: str = "cpu",
    output_base: str = "runs",
) -> "Logger":  # noqa: F821
    """Train a single model and return the Logger instance."""
    from mini_networks.api.dependencies import get_model_registry
    from mini_networks.core.logging.logger import Logger

    registry = get_model_registry()
    if model not in registry:
        raise ValueError(f"Unknown model: {model!r}. Available: {MODELS}")

    ConfigClass, TrainerClass, dataloader_fn = registry[model]
    config = ConfigClass(
        epochs=epochs,
        batch_size=batch_size,
        fast_demo=fast_demo,
        data_root=data_root,
        device=device,
    )
    ts = _ts()
    output_dir = os.path.join(output_base, model, ts)
    logger = Logger(output_dir=output_dir, run_name=ts)

    dataloader = dataloader_fn(config, split="train")
    trainer = TrainerClass()

    console.print(Panel(
        f"[bold]{_DESCRIPTIONS.get(model, model)}[/bold]\n"
        f"[dim]fast_demo={fast_demo}  epochs={epochs}  device={device}[/dim]\n"
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
    data_root: str = "/tmp/mini_networks_data",
    device: str = "cpu",
    output_base: str = "runs",
) -> dict:
    """Run a cross-model composition pipeline."""
    if composition not in COMPOSITIONS:
        raise ValueError(f"Unknown composition: {composition!r}. Available: {COMPOSITIONS}")

    console.print(Panel(
        f"[bold]{_DESCRIPTIONS.get(composition, composition)}[/bold]\n"
        f"[dim]fast_demo={fast_demo}  device={device}[/dim]",
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
            result = _run_clip_guided_diffusion(fast_demo, data_root, device)
        elif composition == "transformer_clip_diffusion":
            result = _run_transformer_clip_diffusion(fast_demo, data_root, device)
        elif composition == "gan_diffusion_comparison":
            result = _run_gan_diffusion_comparison(fast_demo, data_root, device)
        elif composition == "clip_guided_gan":
            result = _run_clip_guided_gan(fast_demo, data_root, device)
        elif composition == "classifier_guided_diffusion":
            result = _run_classifier_guided_diffusion(fast_demo, data_root, device)
        elif composition == "rag_guided_generation":
            result = _run_rag_guided_generation(fast_demo, data_root, device)
        elif composition == "lora_lm":
            result = _run_lora_lm(fast_demo, data_root, device)
        elif composition == "segment_then_detect":
            result = _run_segment_then_detect(fast_demo, data_root, device)
        elif composition == "multitask_vision":
            result = _run_multitask_vision(fast_demo, data_root, device)
        elif composition == "diffusion_distillation":
            result = _run_diffusion_distillation(fast_demo, data_root, device)
        elif composition == "audio_text_contrastive":
            result = _run_audio_text_contrastive(fast_demo, data_root, device)
        elif composition == "tabular_text_cross_attention":
            result = _run_tabular_text_cross_attention(fast_demo, data_root, device)
        elif composition == "audio_text_dual_encoder":
            result = _run_audio_text_dual_encoder(fast_demo, data_root, device)
        elif composition == "tabular_text_dual_encoder":
            result = _run_tabular_text_dual_encoder(fast_demo, data_root, device)
        elif composition == "classifier_guided_gan":
            result = _run_classifier_guided_gan(fast_demo, data_root, device)
        elif composition == "rag_conditioned_diffusion":
            result = _run_rag_conditioned_diffusion(fast_demo, data_root, device)
        elif composition == "image_captioning":
            result = _run_image_captioning(fast_demo, data_root, device)
        elif composition == "multimodal_fusion_baseline":
            result = _run_multimodal_fusion_baseline(fast_demo, data_root, device)
        elif composition == "latent_diffusion":
            result = _run_latent_diffusion(fast_demo, data_root, device)
        else:
            raise RuntimeError(f"No runner implemented for {composition!r}")

    console.print("[green]Composition complete.[/green]")
    return result


def _run_clip_guided_diffusion(fast_demo, data_root, device) -> dict:
    from mini_networks.compositions.clip_guided_diffusion import (
        CLIPGuidedDiffusion,
        CLIPGuidedDiffusionConfig,
    )
    import tempfile
    from mini_networks.core.logging.logger import Logger

    cfg = CLIPGuidedDiffusionConfig(fast_demo=fast_demo, data_root=data_root, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="clip_guided")
        pipeline = CLIPGuidedDiffusion()
        pipeline.train(cfg, logger)
        images = pipeline.text_to_image("digit zero", cfg)
        console.print(f"  Generated images shape: [cyan]{images.shape}[/cyan]")
        return {"images": images, "config": cfg}


def _run_transformer_clip_diffusion(fast_demo, data_root, device) -> dict:
    from mini_networks.compositions.transformer_clip_diffusion import (
        TransformerCLIPDiffusion,
        TransformerCLIPDiffusionConfig,
    )
    import tempfile
    from mini_networks.core.logging.logger import Logger

    cfg = TransformerCLIPDiffusionConfig(fast_demo=fast_demo, data_root=data_root, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="tcd")
        pipeline = TransformerCLIPDiffusion()
        pipeline.train(cfg, logger)
        images, class_id, prompts = pipeline.generate_image("KING", cfg)
        console.print(f"  Best class: [cyan]{class_id}[/cyan]  Generated shape: [cyan]{images.shape}[/cyan]")
        return {"images": images, "class_id": class_id, "prompts": prompts}


def _run_gan_diffusion_comparison(fast_demo, data_root, device) -> dict:
    from mini_networks.compositions.gan_diffusion_comparison import (
        GANDiffusionComparison,
        GANDiffusionConfig,
    )
    import tempfile
    from mini_networks.core.logging.logger import Logger

    cfg = GANDiffusionConfig(fast_demo=fast_demo, data_root=data_root, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="gdc")
        cmp = GANDiffusionComparison()
        cmp.train(cfg, logger)
        results = cmp.compare(cfg, n_samples=4)
        console.print(
            f"  GAN diversity: [cyan]{results['gan_diversity']:.4f}[/cyan]  "
            f"Diffusion diversity: [cyan]{results['diffusion_diversity']:.4f}[/cyan]"
        )
        return results


def _run_clip_guided_gan(fast_demo, data_root, device) -> dict:
    from mini_networks.compositions.clip_guided_gan import CLIPGuidedGAN, CLIPGuidedGANConfig
    import tempfile
    from mini_networks.core.logging.logger import Logger

    cfg = CLIPGuidedGANConfig(fast_demo=fast_demo, data_root=data_root, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="clip_guided_gan")
        pipeline = CLIPGuidedGAN()
        pipeline.train(cfg, logger)
        images = pipeline.sample(cfg, n=4)
        console.print(f"  Generated images shape: [cyan]{images.shape}[/cyan]")
        return {"images": images, "config": cfg}


def _run_classifier_guided_diffusion(fast_demo, data_root, device) -> dict:
    from mini_networks.compositions.classifier_guided_diffusion import (
        ClassifierGuidedDiffusion,
        ClassifierGuidedDiffusionConfig,
    )
    import tempfile
    from mini_networks.core.logging.logger import Logger

    cfg = ClassifierGuidedDiffusionConfig(fast_demo=fast_demo, data_root=data_root, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="cls_guided_diff")
        pipeline = ClassifierGuidedDiffusion()
        pipeline.run(cfg, logger)
        images = pipeline.sample(cfg, n=4)
        console.print(f"  Generated images shape: [cyan]{images.shape}[/cyan]")
        return {"images": images, "config": cfg}


def _run_rag_guided_generation(fast_demo, data_root, device) -> dict:
    from mini_networks.compositions.rag_guided_generation import (
        RAGGuidedGeneration,
        RAGGuidedGenerationConfig,
    )
    import tempfile
    from mini_networks.core.logging.logger import Logger

    cfg = RAGGuidedGenerationConfig(fast_demo=fast_demo, data_root=data_root, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="rag_guided")
        pipeline = RAGGuidedGeneration()
        pipeline.train(cfg, logger)
        text = pipeline.generate(cfg, "To be or not to be", max_new_tokens=24)
        console.print(f"  Sample: [cyan]{text[:120]}[/cyan]")
        return {"text": text, "config": cfg}


def _run_lora_lm(fast_demo, data_root, device) -> dict:
    from mini_networks.compositions.lora_lm import LoRALM, LoRALMConfig
    import tempfile
    from mini_networks.core.logging.logger import Logger

    cfg = LoRALMConfig(fast_demo=fast_demo, data_root=data_root, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="lora_lm")
        pipeline = LoRALM()
        pipeline.train(cfg, logger)
        text = pipeline.generate(cfg, "Hello", max_new_tokens=16)
        console.print(f"  Sample: [cyan]{text[:120]}[/cyan]")
        return {"text": text, "config": cfg}


def _run_segment_then_detect(fast_demo, data_root, device) -> dict:
    from mini_networks.compositions.segment_then_detect import SegmentThenDetect, SegmentThenDetectConfig
    from mini_networks.core.data.registry import get_dataloader
    import tempfile
    from mini_networks.core.logging.logger import Logger

    cfg = SegmentThenDetectConfig(fast_demo=fast_demo, data_root=data_root, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="seg_then_det")
        pipeline = SegmentThenDetect()
        pipeline.train(cfg, logger)
        dl = get_dataloader(
            name=cfg.dataset,
            data_root=cfg.data_root,
            split="train",
            task="classification",
            batch_size=4,
            fast_demo=True,
        )
        images, _ = next(iter(dl))
        bboxes = pipeline.infer_bbox(cfg, images)
        console.print(f"  BBoxes shape: [cyan]{bboxes.shape}[/cyan]")
        return {"bboxes": bboxes, "config": cfg}


def _run_multitask_vision(fast_demo, data_root, device) -> dict:
    from mini_networks.compositions.multitask_vision import MultiTaskVision, MultiTaskVisionConfig
    import tempfile
    from mini_networks.core.logging.logger import Logger

    cfg = MultiTaskVisionConfig(fast_demo=fast_demo, data_root=data_root, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="multitask_vision")
        pipeline = MultiTaskVision()
        pipeline.train(cfg, logger)
        return {"config": cfg}


def _run_diffusion_distillation(fast_demo, data_root, device) -> dict:
    from mini_networks.compositions.diffusion_distillation import (
        DiffusionDistillation,
        DiffusionDistillationConfig,
    )
    import tempfile
    from mini_networks.core.logging.logger import Logger

    cfg = DiffusionDistillationConfig(fast_demo=fast_demo, data_root=data_root, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="diff_distill")
        pipeline = DiffusionDistillation()
        pipeline.train(cfg, logger)
        return {"config": cfg}


def _run_audio_text_contrastive(fast_demo, data_root, device) -> dict:
    from mini_networks.compositions.audio_text_contrastive import (
        AudioTextContrastive,
        AudioTextContrastiveConfig,
    )
    import tempfile
    from mini_networks.core.logging.logger import Logger

    cfg = AudioTextContrastiveConfig(fast_demo=fast_demo, data_root=data_root, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="audio_text_contrastive")
        pipeline = AudioTextContrastive()
        pipeline.train(cfg, logger)
        return {"config": cfg}


def _run_tabular_text_cross_attention(fast_demo, data_root, device) -> dict:
    from mini_networks.compositions.tabular_text_cross_attention import (
        TabularTextCrossAttention,
        TabularTextCrossAttentionConfig,
    )
    import tempfile
    from mini_networks.core.logging.logger import Logger

    cfg = TabularTextCrossAttentionConfig(fast_demo=fast_demo, data_root=data_root, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="tabular_text_cross_attention")
        pipeline = TabularTextCrossAttention()
        pipeline.train(cfg, logger)
        return {"config": cfg}


def _run_audio_text_dual_encoder(fast_demo, data_root, device) -> dict:
    from mini_networks.compositions.audio_text_dual_encoder import (
        AudioTextDualEncoder,
        AudioTextDualEncoderConfig,
    )
    import tempfile
    from mini_networks.core.logging.logger import Logger

    cfg = AudioTextDualEncoderConfig(fast_demo=fast_demo, data_root=data_root, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="audio_text_dual")
        pipeline = AudioTextDualEncoder()
        pipeline.train(cfg, logger)
        return {"config": cfg}


def _run_tabular_text_dual_encoder(fast_demo, data_root, device) -> dict:
    from mini_networks.compositions.tabular_text_dual_encoder import (
        TabularTextDualEncoder,
        TabularTextDualEncoderConfig,
    )
    import tempfile
    from mini_networks.core.logging.logger import Logger

    cfg = TabularTextDualEncoderConfig(fast_demo=fast_demo, data_root=data_root, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="tabular_text_dual")
        pipeline = TabularTextDualEncoder()
        pipeline.train(cfg, logger)
        return {"config": cfg}


def _run_classifier_guided_gan(fast_demo, data_root, device) -> dict:
    from mini_networks.compositions.classifier_guided_gan import (
        ClassifierGuidedGAN,
        ClassifierGuidedGANConfig,
    )
    import tempfile
    from mini_networks.core.logging.logger import Logger

    cfg = ClassifierGuidedGANConfig(fast_demo=fast_demo, data_root=data_root, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="classifier_guided_gan")
        pipeline = ClassifierGuidedGAN()
        pipeline.train(cfg, logger)
        return {"config": cfg}


def _run_rag_conditioned_diffusion(fast_demo, data_root, device) -> dict:
    from mini_networks.compositions.rag_conditioned_diffusion import (
        RAGConditionedDiffusion,
        RAGConditionedDiffusionConfig,
    )
    import tempfile
    from mini_networks.core.logging.logger import Logger

    cfg = RAGConditionedDiffusionConfig(fast_demo=fast_demo, data_root=data_root, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="rag_conditioned_diffusion")
        pipeline = RAGConditionedDiffusion()
        pipeline.train(cfg, logger)
        images, prompt = pipeline.sample(cfg)
        console.print(f"  Prompt: [cyan]{prompt[:80]}[/cyan]")
        console.print(f"  Images: [cyan]{images.shape}[/cyan]")
        return {"images": images, "prompt": prompt, "config": cfg}


def _run_image_captioning(fast_demo, data_root, device) -> dict:
    from mini_networks.compositions.image_captioning import ImageCaptioning, ImageCaptioningConfig
    import tempfile
    from mini_networks.core.logging.logger import Logger

    cfg = ImageCaptioningConfig(fast_demo=fast_demo, data_root=data_root, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="image_captioning")
        pipeline = ImageCaptioning()
        pipeline.train(cfg, logger)
        return {"config": cfg}


def _run_multimodal_fusion_baseline(fast_demo, data_root, device) -> dict:
    from mini_networks.compositions.multimodal_fusion_baseline import (
        MultimodalFusionBaseline,
        MultimodalFusionConfig,
    )
    import tempfile
    from mini_networks.core.logging.logger import Logger

    cfg = MultimodalFusionConfig(fast_demo=fast_demo, data_root=data_root, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="multimodal_fusion")
        pipeline = MultimodalFusionBaseline()
        pipeline.train(cfg, logger)
        return {"config": cfg}


def _run_latent_diffusion(fast_demo, data_root, device) -> dict:
    from mini_networks.compositions.latent_diffusion import LatentDiffusion, LatentDiffusionConfig
    import tempfile
    from mini_networks.core.logging.logger import Logger

    cfg = LatentDiffusionConfig(fast_demo=fast_demo, data_root=data_root, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(output_dir=tmpdir, run_name="latent_diffusion")
        pipeline = LatentDiffusion()
        pipeline.train(cfg, logger)
        images = pipeline.sample(cfg, n=4)
        console.print(f"  Images: [cyan]{images.shape}[/cyan]")
        return {"images": images, "config": cfg}


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

    fast_raw = console.input("[bold]Fast demo?[/bold] (Y/n): ").strip().lower()
    fast_demo = fast_raw not in ("n", "no")
    device = console.input("[bold]Device[/bold] [cpu/cuda/mps] (default: cpu): ").strip() or "cpu"

    if choice == "1":
        run_model(model, fast_demo=fast_demo, device=device)
    else:
        run_composition(comp, fast_demo=fast_demo, device=device)


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
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--data_root", default="/tmp/mini_networks_data")

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
            data_root=args.data_root,
            device=args.device,
        )
    else:
        run_composition(
            args.composition,
            fast_demo=not args.no_fast,
            data_root=args.data_root,
            device=args.device,
        )
