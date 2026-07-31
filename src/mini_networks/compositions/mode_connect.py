"""Mode connectivity: a low-loss simplex between independent solutions.

Key idea (Benton et al., "Loss Surface Simplexes" / SPRO, arXiv 2102.13042):
two networks trained from different seeds land in different loss-basin
"modes" — but those modes are not isolated points. They are connected by
low-loss volumes: train ONE extra vertex (initialized at the midpoint) so
that every convex combination of the three weight vectors has low loss, and
you get a whole 2-simplex of working networks. Sampling networks from inside
that simplex gives a cheap ensemble that beats any single vertex.

This composition:
  1. trains two SmallCNN classifiers from different seeds (frozen vertices),
  2. learns a third vertex by minimizing loss at Dirichlet-sampled convex
     combinations of the three weight sets (torch.func.functional_call —
     the template module never owns the merged weights),
  3. compares single-model accuracy vs a simplex-ensemble (average softmax
     over interior samples),
  4. renders the val-loss heatmap over the simplex plane as
     artifacts/loss_surface.png — the paper's signature visual: bright
     (low-loss) everywhere inside the triangle, not just at the corners.

Deliberately simplified vs the paper: one extra vertex (a 2-simplex, not the
multi-vertex complexes), linear merge of BatchNorm running stats (fine at
MNIST scale), and a fixed eval subset for the heatmap.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torchvision.utils import save_image

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import make_classification_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.models.classifier.config import ClassifierConfig
from mini_networks.models.classifier.model import SmallCNN

import logging

log = logging.getLogger(__name__)


class ModeConnectConfig(BaseConfig):
    model_name: str = "mode_connect"
    dataset: str = "mnist"
    hidden_dim: int = 32
    num_classes: int = 10
    seeds: tuple[int, int] = (0, 1)
    n_grid: int = 15       # loss-surface resolution (n_grid x n_grid barycentric raster)
    n_ensemble: int = 5    # interior samples averaged for the ensemble prediction


class ModeConnect:
    """Two frozen vertices + one learned vertex spanning a low-loss simplex."""

    def _clf_config(self, config: ModeConnectConfig) -> ClassifierConfig:
        return ClassifierConfig(
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            dataset=config.dataset,
            data_root=config.data_root,
            device=config.device,
            fast_demo=config.fast_demo,
            training_tier=config.training_tier,
            batch_size=config.batch_size,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
        )

    @staticmethod
    def _merge(vertices: list[dict], weights) -> dict:
        merged = {}
        for k in vertices[0]:
            if vertices[0][k].is_floating_point():
                merged[k] = sum(w * v[k] for w, v in zip(weights, vertices))
            else:
                merged[k] = vertices[0][k]  # int buffers (num_batches_tracked)
        return merged

    def _train_vertex(self, model, dl, config, logger: Logger, key: str) -> None:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        for epoch in range(config.effective_epochs):
            total = 0.0
            for x, y in dl:
                x, y = x.to(config.device), y.to(config.device)
                loss = F.cross_entropy(model(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            logger.log_metrics(epoch, {key: total / max(1, len(dl)), "loss": total / max(1, len(dl))})

    @torch.no_grad()
    def _accuracy(self, template, params: dict, dl, device, n_batches: int | None = None) -> float:
        correct = total = 0
        for i, (x, y) in enumerate(dl):
            if n_batches is not None and i >= n_batches:
                break
            logits = functional_call(template, params, (x.to(device),))
            correct += (logits.argmax(-1).cpu() == y).sum().item()
            total += y.numel()
        return correct / max(1, total)

    def train_all(self, config: ModeConnectConfig, logger: Logger) -> dict:
        device = config.device
        cfgc = self._clf_config(config)
        dl_train = make_classification_dataloader(cfgc, split="train")
        try:
            dl_val = make_classification_dataloader(cfgc, split="test")
        except Exception:
            dl_val = make_classification_dataloader(cfgc, split="train")

        # 1) two independent modes
        vertices: list[dict] = []
        for i, seed in enumerate(config.seeds):
            torch.manual_seed(seed)
            model = SmallCNN(config.hidden_dim, config.num_classes).to(device)
            log.info("training vertex %d (seed %d)", i, seed)
            self._train_vertex(model, dl_train, config, logger, key=f"clf_{'ab'[i]}_loss")
            vertices.append({k: v.detach().clone() for k, v in model.state_dict().items()})

        # 2) learned third vertex, init at the midpoint
        template = SmallCNN(config.hidden_dim, config.num_classes).to(device)
        template.eval()  # BN in eval mode: merged running stats are used, not mutated
        param_names = {k for k, _ in template.named_parameters()}
        theta3 = {}
        for k in vertices[0]:
            t = ((vertices[0][k] + vertices[1][k]) / 2).clone()
            if k in param_names:  # buffers (BN running stats) stay non-trainable
                t.requires_grad_(True)
            theta3[k] = t
        trainable = [t for t in theta3.values() if t.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=config.learning_rate)
        dirichlet = torch.distributions.Dirichlet(torch.ones(3))
        for epoch in range(config.effective_epochs):
            total = 0.0
            for x, y in dl_train:
                x, y = x.to(device), y.to(device)
                w = dirichlet.sample().to(device)
                merged = self._merge([vertices[0], vertices[1], theta3], w)
                loss = F.cross_entropy(functional_call(template, merged, (x,)), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            logger.log_metrics(epoch, {"simplex_loss": total / max(1, len(dl_train))})
        theta3 = {k: v.detach() for k, v in theta3.items()}

        # 3) single vs simplex-ensemble accuracy
        single_acc = self._accuracy(template, vertices[0], dl_val, device)
        probs_sum, targets = None, []
        with torch.no_grad():
            xs, ys = [], []
            for i, (x, y) in enumerate(dl_val):
                if i >= (config.max_eval_batches or 8):
                    break
                xs.append(x)
                ys.append(y)
            x_all = torch.cat(xs).to(device)
            y_all = torch.cat(ys)
            for _ in range(config.n_ensemble):
                w = dirichlet.sample().to(device)
                merged = self._merge([vertices[0], vertices[1], theta3], w)
                p = F.softmax(functional_call(template, merged, (x_all,)), dim=-1)
                probs_sum = p if probs_sum is None else probs_sum + p
            ensemble_acc = (probs_sum.argmax(-1).cpu() == y_all).float().mean().item()

        # 4) loss-surface heatmap over the simplex plane
        n = config.n_grid
        surface = torch.full((n, n), float("nan"))
        with torch.no_grad():
            for i in range(n):
                for j in range(n - i):
                    w = torch.tensor(
                        [i / (n - 1), j / (n - 1), max(0.0, 1 - (i + j) / (n - 1))],
                        device=device,
                    )
                    w = w / w.sum()
                    merged = self._merge([vertices[0], vertices[1], theta3], w)
                    loss = F.cross_entropy(
                        functional_call(template, merged, (x_all,)), y_all.to(device)
                    )
                    surface[i, j] = loss.item()
        valid = surface[~surface.isnan()]
        norm = (surface - valid.min()) / (valid.max() - valid.min() + 1e-9)
        norm = torch.nan_to_num(norm, nan=1.0)  # outside the triangle: bright
        img = (1.0 - norm).clamp(0, 1).unsqueeze(0)  # dark = high loss, light = low
        save_image(img, str(logger.artifact_path("loss_surface.png")))

        torch.save({"vertex_a": vertices[0], "vertex_b": vertices[1], "vertex_c": theta3},
                   logger.artifact_path("simplex_vertices.pt"))
        return {
            "single_accuracy": single_acc,
            "ensemble_accuracy": ensemble_acc,
            "run_dir": str(logger.run_dir),
        }
