"""Per-item inference showcases: human-viewable evidence that an item learned.

The gate's probe validates output SHAPE; a showcase saves output CONTENT —
sample grids for generative models, pred-vs-true tables for classifiers,
generated text for LMs, 1-NN label accuracy for embedding models, plus any
image/text artifacts the run itself produced. Files land under
``<showcase_dir>/<item>/``; the cloud sweep uploads them to
``gs://…/sweeps/<id>/samples/<item>/`` and ``main.py sweep-samples`` pulls a
whole sweep's showcases to a local folder (e.g. ~/Downloads).

Every section is best-effort: a showcase must never fail the gate, so errors
become lines in summary.txt instead of exceptions.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_GRID_MAX = 16
_TEXT_PROMPT = "To be, or not to be"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif"}

VISION_CLS = {"classifier", "resnet", "vit", "mobilenet", "convnext", "lora"}
EMBEDDERS = {"simclr", "dino", "vision_embed", "clip"}
SAMPLERS = {"diffusion", "gan", "pixelcnn", "vae", "tabular_diffusion"}
TEXT_LMS = {"transformer", "mamba", "rnn", "rlhf"}


def _save_grid(tensor, path: Path) -> bool:
    import torch
    from torchvision.utils import save_image

    t = tensor.detach().float().cpu()
    if t.dim() == 3:  # [B,H,W] -> add channel
        t = t.unsqueeze(1)
    if t.dim() != 4 or t.size(1) not in (1, 3) or t.size(-1) < 8 or t.size(-2) < 8:
        return False
    save_image(t[:_GRID_MAX], str(path), nrow=4, normalize=True)
    return True


def _dump_generic(output: Any, dest: Path, lines: list[str], prefix: str = "") -> None:
    """Tensors that look like images -> PNG grids; everything else -> summary lines."""
    import torch

    if isinstance(output, dict):
        for key, value in output.items():
            _dump_generic(value, dest, lines, prefix=f"{prefix}{key}" if not prefix else f"{prefix}.{key}")
        return
    name = prefix or "output"
    if isinstance(output, torch.Tensor):
        if _save_grid(output, dest / f"{name}.png"):
            lines.append(f"{name}: image grid -> {name}.png {tuple(output.shape)}")
        else:
            flat = output.detach().float().flatten()
            lines.append(f"{name}: tensor{tuple(output.shape)} head={[round(v, 4) for v in flat[:8].tolist()]}")
    elif isinstance(output, str):
        lines.append(f"{name}: {output}")
    elif isinstance(output, (list, tuple)):
        lines.append(f"{name}: {str(output)[:400]}")
    elif output is not None:
        lines.append(f"{name}: {output!r}")


def _neighbor_grid(trainer, config, dataloader_fn, dest: Path, lines: list[str],
                   n_anchors: int = 6, n_neighbors: int = 5, max_samples: int = 512) -> None:
    """Visual proof for embedding models: each row = an anchor image followed by
    its nearest neighbors in embedding space. Same-digit rows = it learned."""
    import torch
    from torchvision.utils import save_image

    try:
        dl = dataloader_fn(config, split="test")
    except Exception:
        dl = dataloader_fn(config, split="train")
    images, embeds = [], []
    with torch.no_grad():
        for batch in dl:
            x = batch[0]
            out = trainer.infer(config, {"images": x})
            e = out.get("embeddings", out.get("image_embeds")) if isinstance(out, dict) else out
            if e is None:
                return
            images.append(x.detach().cpu())
            embeds.append(e.detach().float().cpu())
            if sum(t.size(0) for t in images) >= max_samples:
                break
    x = torch.cat(images)[:max_samples]
    e = torch.nn.functional.normalize(torch.cat(embeds)[:max_samples], dim=-1)
    sim = e @ e.T
    sim.fill_diagonal_(-2.0)
    rows = []
    for a in range(0, n_anchors):
        idx = sim[a].topk(n_neighbors).indices
        rows.append(torch.cat([x[a : a + 1], x[idx]]))
    grid = torch.cat(rows)
    save_image(grid, str(dest / "neighbors.png"), nrow=n_neighbors + 1, normalize=True)
    lines.append(
        f"neighbors.png: {n_anchors} rows of [anchor | {n_neighbors} nearest in embedding space] — "
        "rows of the same digit mean the embedding learned"
    )


def _spectrogram_grid(waves, dest: Path, lines: list[str], n: int = 8) -> None:
    """Waveforms -> log-magnitude STFT images (audio's natural visualization)."""
    import torch
    from torchvision.utils import save_image

    w = waves[:n].detach().float().cpu()
    if w.dim() == 3:  # [B,1,T] -> [B,T]
        w = w.squeeze(1)
    spec = torch.stft(w, n_fft=128, hop_length=32, return_complex=True,
                      window=torch.hann_window(128))
    mag = spec.abs().clamp_min(1e-6).log()
    mag = (mag - mag.amin(dim=(1, 2), keepdim=True)) / (
        mag.amax(dim=(1, 2), keepdim=True) - mag.amin(dim=(1, 2), keepdim=True) + 1e-6)
    save_image(mag.flip(1).unsqueeze(1), str(dest / "spectrograms.png"), nrow=4)
    lines.append(f"spectrograms.png: log-STFT of the {mag.size(0)} input clips (order matches pred/true)")


def _render_maze_path(trainer, config, dest: Path, lines: list[str]) -> None:
    """Roll the greedy policy and paint the maze + trajectory as an image."""
    import torch
    from torchvision.utils import save_image

    env = getattr(trainer, "env", None)
    agent = getattr(trainer, "agent", None) or getattr(trainer, "policy", None)
    if env is None or agent is None:
        return
    colors = {  # HOLE, PATH, START, GOAL
        0: (0.16, 0.12, 0.20), 1: (0.99, 0.96, 0.86), 2: (0.49, 0.37, 1.0), 3: (1.0, 0.82, 0.25),
    }
    h, w = env.maze.shape
    img = torch.zeros(3, h, w)
    for r in range(h):
        for c in range(w):
            for ch, v in enumerate(colors[int(env.maze[r, c])]):
                img[ch, r, c] = v
    state = env.reset()
    done, steps = False, 0
    trail = [env.agent_pos]
    saved_eps = getattr(agent, "epsilon", None)
    if saved_eps is not None:
        agent.epsilon = 0.0  # greedy rollout — exploration noise ruins the render
    while not done and steps < 200:
        if hasattr(agent, "act"):
            action = agent.act(state)
        else:  # REINFORCE policy net: greedy argmax
            with torch.no_grad():
                action = int(agent(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)).argmax())
        state, reward, done, _ = env.step(action)
        trail.append(env.agent_pos)
        steps += 1
    if saved_eps is not None:
        agent.epsilon = saved_eps
    for i, (r, c) in enumerate(trail[:-1]):
        t = 0.35 + 0.5 * i / max(1, len(trail) - 1)
        img[:, r, c] = torch.tensor([0.36 * t, 0.65 * t, 0.28 * t]) + img[:, r, c] * 0.25
    img[:, trail[-1][0], trail[-1][1]] = torch.tensor([0.88, 0.33, 0.25])
    big = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=32, mode="nearest")[0]
    save_image(big, str(dest / "maze_path.png"))
    lines.append(
        f"maze_path.png: purple=start gold=goal green=greedy path (light→dark = time) red=final pos; "
        f"{steps} steps, reached_goal={bool(reward == 1.0)}"
    )


def _knn1_accuracy(trainer, config, dataloader_fn, max_samples: int = 512) -> str:
    """1-NN label accuracy on embeddings — the legible 'did it learn?' number
    for self-supervised models (random embeddings score ~chance)."""
    import torch

    try:
        dl = dataloader_fn(config, split="test")
    except Exception:
        dl = dataloader_fn(config, split="train")
    embeds, labels = [], []
    with torch.no_grad():
        for batch in dl:
            x, y = batch[0], batch[-1]
            out = trainer.infer(config, {"images": x})
            e = out.get("embeddings", out.get("image_embeds")) if isinstance(out, dict) else out
            if e is None:
                return "knn1_accuracy: n/a (no embeddings in infer output)"
            embeds.append(e.detach().float().cpu())
            labels.append(torch.as_tensor(y).flatten())
            if sum(t.size(0) for t in embeds) >= max_samples:
                break
    e = torch.cat(embeds)[:max_samples]
    y = torch.cat(labels)[:max_samples]
    e = torch.nn.functional.normalize(e, dim=-1)
    sim = e @ e.T
    sim.fill_diagonal_(-2.0)  # exclude self-match
    acc = (y[sim.argmax(dim=1)] == y).float().mean().item()
    return f"knn1_accuracy: {acc:.4f} over {e.size(0)} samples (chance ~0.1)"


def _pred_vs_true(out: Any, targets, lines: list[str]) -> None:
    import torch

    preds = out.get("predictions") if isinstance(out, dict) else None
    if preds is None:
        return
    preds = torch.as_tensor(preds).flatten()[:_GRID_MAX]
    truth = torch.as_tensor(targets).flatten()[: preds.numel()]
    acc = (preds == truth).float().mean().item()
    lines.append(f"pred: {preds.tolist()}")
    lines.append(f"true: {truth.tolist()}")
    lines.append(f"batch_accuracy: {acc:.4f}")


def save_model_showcase(name: str, trainer, config, dataloader_fn, dest: str | Path) -> Path:
    import torch

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"item: {name}"]

    batch = None
    try:
        try:
            dl = dataloader_fn(config, split="test")
        except Exception:
            dl = dataloader_fn(config, split="train")
        batch = next(iter(dl))
    except Exception as e:
        lines.append(f"dataloader error: {e}")

    try:
        if name in SAMPLERS:
            out = trainer.infer(config, {"n_samples": _GRID_MAX, "seed": config.seed, "sample": _GRID_MAX})
            _dump_generic(out, dest, lines)
        elif name in EMBEDDERS:
            lines.append(_knn1_accuracy(trainer, config, dataloader_fn))
            if name != "clip":  # clip's loader yields (image, text) pairs, not two views
                _neighbor_grid(trainer, config, dataloader_fn, dest, lines)
        elif name in VISION_CLS and batch is not None:
            x = batch[0][:_GRID_MAX]
            if _save_grid(x, dest / "inputs.png"):
                lines.append("inputs: inputs.png")
            out = trainer.infer(config, {"images": x} if name == "lora" else x)
            _pred_vs_true(out, batch[1], lines)
        elif name in TEXT_LMS:
            out = trainer.infer(config, {"prompt": _TEXT_PROMPT, "max_new_tokens": 120})
            lines.append(f"prompt: {_TEXT_PROMPT}")
            _dump_generic(out, dest, lines)
        elif name == "rag":
            out = trainer.infer(config, {"query": _TEXT_PROMPT, "max_new_tokens": 120})
            lines.append(f"query: {_TEXT_PROMPT}")
            _dump_generic(out, dest, lines)
        elif name == "text_seq2seq" and batch is not None:
            out = trainer.infer(config, {"src": batch[0][0]})
            lines.append(f"src ids: {batch[0][0].flatten()[:32].tolist()}")
            _dump_generic(out, dest, lines)
        elif name == "text_token_classifier" and batch is not None:
            out = trainer.infer(config, {"tokens": batch[0][0]})
            _dump_generic(out, dest, lines)
        elif name == "tabular_classifier" and batch is not None:
            out = trainer.infer(config, {"features": batch[0][:_GRID_MAX]})
            _pred_vs_true(out, batch[1], lines)
        elif name in {"segmentation", "detection", "unet_ae"} and batch is not None:
            x = batch[0][:8]
            if _save_grid(x, dest / "inputs.png"):
                lines.append("inputs: inputs.png")
            out = trainer.infer(config, {"images": x} if name != "unet_ae" else x)
            _dump_generic(out, dest, lines)
        elif name.startswith("audio_") and batch is not None:
            out = trainer.infer(config, batch[0][:_GRID_MAX])
            _pred_vs_true(out, batch[1], lines)
            if batch[0].dim() >= 4:  # loader already yields spectrogram images
                if _save_grid(batch[0][:8], dest / "spectrograms.png"):
                    lines.append("spectrograms.png: the 8 input spectrograms (order matches pred/true)")
            else:  # raw waveforms — render our own log-STFT
                _spectrogram_grid(batch[0], dest, lines)
        elif name in {"rl_maze", "reinforce"}:
            out = trainer.infer(config, {})
            _dump_generic(out, dest, lines)
            _render_maze_path(trainer, config, dest, lines)
        elif batch is not None:  # fallback: feed the batch, dump whatever comes out
            out = trainer.infer(config, batch[0][:_GRID_MAX])
            _dump_generic(out, dest, lines)
    except Exception as e:
        lines.append(f"showcase error: {type(e).__name__}: {e}")

    (dest / "summary.txt").write_text("\n".join(lines) + "\n")
    return dest


def save_composition_showcase(name: str, output: Any, run_dir: str | None, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"item: {name}"]
    try:
        if isinstance(output, dict):
            payload = {k: v for k, v in output.items() if k not in {"config", "run_dir"}}
            _dump_generic(payload, dest, lines)
    except Exception as e:
        lines.append(f"showcase error: {type(e).__name__}: {e}")

    # Compositions usually save their best visuals during training — copy them.
    if run_dir:
        artifacts = Path(run_dir) / "artifacts"
        if artifacts.is_dir():
            copied = 0
            for f in sorted(artifacts.iterdir()):
                if f.suffix.lower() in _IMAGE_SUFFIXES | {".txt"} and copied < 20:
                    shutil.copy2(f, dest / f.name)
                    copied += 1
            if copied:
                lines.append(f"copied {copied} training artifact(s) from {artifacts}")

    (dest / "summary.txt").write_text("\n".join(lines) + "\n")
    return dest
