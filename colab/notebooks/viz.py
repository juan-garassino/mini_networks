"""Shared visualization helpers for mini_networks Colab notebooks.

All functions are designed to work inside Jupyter/Colab without extra imports
beyond matplotlib and torchvision — both available in any standard Colab env.

Usage::

    from viz import plot_metrics, show_image_grid, show_text_sample, show_maze
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Metric curves
# ---------------------------------------------------------------------------

def plot_metrics(logger, keys: list[str] | None = None, figsize: tuple = (12, 4)) -> None:
    """Plot training curves from a Logger or a path to metrics.jsonl.

    Args:
        logger: a ``mini_networks.core.logging.logger.Logger`` instance,
                or a str/Path pointing to a ``metrics.jsonl`` file.
        keys:   list of metric keys to plot; None plots all numeric keys.
        figsize: matplotlib figure size.
    """
    import json
    import matplotlib.pyplot as plt

    # Load records
    if isinstance(logger, (str, Path)):
        path = Path(logger)
        records: list[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        records = logger.read_metrics()

    if not records:
        print("No metrics found.")
        return

    # Group by key
    from collections import defaultdict
    series: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for r in records:
        k = r.get("key", "")
        v = r.get("value")
        step = r.get("step", 0)
        if isinstance(v, (int, float)):
            series[k][0].append(step)
            series[k][1].append(v)

    if keys is not None:
        series = {k: v for k, v in series.items() if k in keys}

    if not series:
        print("No numeric metrics to plot.")
        return

    n = len(series)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0], figsize[1]))
    if n == 1:
        axes = [axes]

    for ax, (key, (steps, values)) in zip(axes, series.items()):
        ax.plot(steps, values, linewidth=1.5)
        ax.set_title(key, fontsize=11)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Image grids
# ---------------------------------------------------------------------------

def show_image_grid(
    tensor,
    title: str = "",
    nrow: int = 4,
    figsize: tuple = (10, 3),
) -> None:
    """Display a batch of images as a grid.

    Args:
        tensor: torch.Tensor of shape [B, C, H, W] or [B, H, W], values in [0,1].
        title:  figure title.
        nrow:   images per row in the grid.
        figsize: matplotlib figure size.
    """
    import matplotlib.pyplot as plt
    import torch

    t = tensor.detach().cpu()
    if t.dim() == 3:
        t = t.unsqueeze(1)  # [B, 1, H, W]

    # Use torchvision if available, fall back to manual grid
    try:
        from torchvision.utils import make_grid
        grid = make_grid(t.clamp(0, 1), nrow=nrow, padding=2)
        img = grid.permute(1, 2, 0).numpy()
        if img.shape[-1] == 1:
            img = img[..., 0]
    except ImportError:
        # Manual fallback: first row only
        img = t[0].permute(1, 2, 0).numpy().squeeze()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    cmap = "gray" if len(img.shape) == 2 else None
    ax.imshow(img, cmap=cmap, interpolation="nearest")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Text samples
# ---------------------------------------------------------------------------

def show_text_sample(text: str, title: str = "Generated text", max_chars: int = 500) -> None:
    """Display a generated text sample with a styled header."""
    display_text = text[:max_chars]
    if len(text) > max_chars:
        display_text += " …"
    border = "─" * 60
    print(f"\n{title}")
    print(border)
    print(display_text)
    print(border + "\n")


# ---------------------------------------------------------------------------
# Maze renderer
# ---------------------------------------------------------------------------

def show_maze(render_str: str, title: str = "Maze") -> None:
    """Pretty-print an ASCII maze (as returned by MazeEnv.render()).

    Legend:  @=agent  S=start  G=goal  X=hole  .=path
    """
    print(f"\n{title}")
    print("+" + "-" * (len(render_str.split("\n")[0])) + "+")
    for row in render_str.split("\n"):
        print(f"| {row} |")
    print("+" + "-" * (len(render_str.split("\n")[0])) + "+\n")


# ---------------------------------------------------------------------------
# CLIP cosine similarity heatmap
# ---------------------------------------------------------------------------

def show_clip_similarity(image_embeds, text_embeds, labels: list[str] | None = None) -> None:
    """Show cosine similarity matrix between image and text embeddings.

    Args:
        image_embeds: torch.Tensor [N, D]
        text_embeds:  torch.Tensor [N, D]
        labels:       optional list of N label strings for axis ticks
    """
    import matplotlib.pyplot as plt
    import torch

    img = torch.nn.functional.normalize(image_embeds.detach().cpu().float(), dim=-1)
    txt = torch.nn.functional.normalize(text_embeds.detach().cpu().float(), dim=-1)
    sim = (img @ txt.T).numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim, cmap="Blues", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="cosine similarity")
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
    ax.set_title("CLIP similarity (image ↔ text)", fontsize=12)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# LoRA weight delta norm bar chart
# ---------------------------------------------------------------------------

def show_lora_deltas(trainer, title: str = "LoRA weight delta norm") -> None:
    """Bar chart of LoRA A/B delta norms for each LoRA layer.

    Args:
        trainer: a trained ``LoRATrainer`` instance with ``trainer.model``.
    """
    import matplotlib.pyplot as plt
    import torch

    model = trainer.model
    if model is None:
        print("No model loaded.")
        return

    names, norms = [], []
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            names.append(name.split(".")[-2] + "." + name.split(".")[-1])
            norms.append(p.detach().cpu().norm().item())

    if not names:
        print("No LoRA parameters found.")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), 3))
    ax.bar(range(len(names)), norms)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("L2 norm")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
