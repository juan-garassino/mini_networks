"""Inference probes: build a minimal infer() input per model and validate that the output is non-degenerate."""
from __future__ import annotations

from typing import Any


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
        "dino",
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
    elif model in {"simclr", "dino", "vision_embed"}:
        output = trainer.infer(config, {"images": batch[0][:1]})
    elif model in {"diffusion", "gan", "pixelcnn", "tabular_diffusion"}:
        output = trainer.infer(config, {"n_samples": 1})
    elif model in {"transformer", "moe", "mamba", "rnn", "rlhf", "grpo", "dpo"}:
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

