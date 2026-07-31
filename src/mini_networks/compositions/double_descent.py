"""Double descent: test error falls AGAIN past the interpolation threshold.

Key idea (framed by Wilson's position paper, arXiv 2503.02113; phenomenon
from Belkin et al. / Nakkiran et al.): the classical U-shaped bias-variance
curve is only the left half of the story. Sweep model capacity on a small,
label-noised training set and test error first falls, then RISES as the model
approaches the interpolation threshold (just enough capacity to memorize the
noisy labels — the worst place to be), then FALLS AGAIN as capacity grows
past it: overparametrized models interpolate the noise *smoothly* instead of
contorting. Wilson's point: this is not deep-learning magic — soft inductive
biases (flexible hypothesis space + preference for simple solutions) explain
it, and the same hump appears in classical model families.

This composition sweeps the width of a small MLP over a subsampled,
label-noised MNIST split (the regime where the hump is visible at zoo
scale), records train/test accuracy per width, and renders the test-error
vs width curve to artifacts/double_descent.png. The deliverable is the
curve; the gate only checks the sweep ran sanely (metric=None).

Deliberately simplified vs the literature: width sweep only (no epoch-wise
double descent), one seed per width, and an MLP rather than a ResNet — the
interpolation threshold just needs to sit inside the swept range, which the
small noisy training set arranges.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataset
from mini_networks.core.logging.logger import Logger

import logging

log = logging.getLogger(__name__)


class DoubleDescentConfig(BaseConfig):
    model_name: str = "double_descent"
    dataset: str = "mnist"
    n_train: int = 1024        # small train set pushes the interpolation threshold left
    label_noise: float = 0.15  # fraction of train labels flipped (the hump needs noise)
    widths: tuple[int, ...] = (2, 4, 8, 16, 48, 128, 512)  # S uses the first 2, M all
    epochs_per_width: int = 40
    num_classes: int = 10


class _LabelNoise(Dataset):
    """Seeded, deterministic label flips on a fixed subset — train split only."""

    def __init__(self, base: Dataset, n_take: int, noise: float, num_classes: int, seed: int = 7):
        g = torch.Generator().manual_seed(seed)
        take = torch.randperm(len(base), generator=g)[:n_take]
        self._items = []
        flip = torch.rand(len(take), generator=g) < noise
        rand_labels = torch.randint(0, num_classes, (len(take),), generator=g)
        for i, idx in enumerate(take.tolist()):
            x, y = base[idx]
            self._items.append((x, int(rand_labels[i]) if flip[i] else int(y)))
        self.n_flipped = int(flip.sum())

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        return self._items[idx]


class _WidthMLP(nn.Module):
    def __init__(self, width: int, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, width),
            nn.ReLU(),
            nn.Linear(width, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _render_curve(widths: list[int], errors: list[float], path, height: int = 128) -> None:
    """Rasterize test-error vs width (log-x) as a simple polyline PNG — no matplotlib."""
    W = height * 2
    img = torch.ones(1, height, W)
    lo, hi = min(errors), max(errors)
    span = (hi - lo) or 1.0
    import math
    xs = [math.log(w) for w in widths]
    x_lo, x_hi = xs[0], xs[-1]
    x_span = (x_hi - x_lo) or 1.0
    pts = [
        (int((x - x_lo) / x_span * (W - 1)), int((1 - (e - lo) / span) * (height - 1)))
        for x, e in zip(xs, errors)
    ]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for s in range(steps + 1):
            x = round(x0 + (x1 - x0) * s / steps)
            y = round(y0 + (y1 - y0) * s / steps)
            img[0, max(0, y - 1):y + 2, max(0, x - 1):x + 2] = 0.0
    save_image(img, str(path))


class DoubleDescent:
    def _widths(self, config: DoubleDescentConfig) -> list[int]:
        if config.effective_tier == "S":
            return list(config.widths[:2])  # smoke: just prove the sweep runs
        return list(config.widths)

    def train_all(self, config: DoubleDescentConfig, logger: Logger) -> dict:
        device = config.device
        base_train = get_dataset(config.dataset, config.data_root, split="train")
        try:
            base_test = get_dataset(config.dataset, config.data_root, split="test")
        except Exception:
            base_test = base_train
        n_train = min(config.n_train, len(base_train)) if not config.effective_fast_demo else 256
        train_ds = _LabelNoise(base_train, n_train, config.label_noise, config.num_classes)
        log.info("train set: %d samples, %d labels flipped", len(train_ds), train_ds.n_flipped)
        dl_train = DataLoader(train_ds, batch_size=128, shuffle=True)
        test_take = min(2048, len(base_test))
        dl_test = DataLoader(
            torch.utils.data.Subset(base_test, range(test_take)), batch_size=256
        )

        widths = self._widths(config)
        epochs = config.limit_steps(config.epochs_per_width, s_cap=1, m_cap=40)
        results = []
        for step, width in enumerate(widths):
            torch.manual_seed(0)  # same init story per width
            model = _WidthMLP(width, config.num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            last_loss = 0.0
            for _ in range(epochs):
                model.train()
                total = 0.0
                for x, y in dl_train:
                    x, y = x.to(device), y.to(device)
                    loss = F.cross_entropy(model(x), y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += loss.item()
                last_loss = total / max(1, len(dl_train))

            @torch.no_grad()
            def _acc(dl):
                model.eval()
                correct = total_n = 0
                for x, y in dl:
                    pred = model(x.to(device)).argmax(-1).cpu()
                    correct += (pred == y).sum().item()
                    total_n += y.numel()
                return correct / max(1, total_n)

            train_acc, test_acc = _acc(dl_train), _acc(dl_test)
            results.append({"width": width, "train_accuracy": train_acc,
                            "test_accuracy": test_acc})
            logger.log_metrics(step, {
                "loss": last_loss, "width": width,
                "train_accuracy": train_acc, "test_accuracy": test_acc,
            })
            log.info("width %4d  train %.3f  test %.3f", width, train_acc, test_acc)

        errors = [1 - r["test_accuracy"] for r in results]
        _render_curve([r["width"] for r in results], errors,
                      logger.artifact_path("double_descent.png"))
        return {
            "widths": [r["width"] for r in results],
            "test_errors": errors,
            "n_flipped": train_ds.n_flipped,
            "run_dir": str(logger.run_dir),
        }
