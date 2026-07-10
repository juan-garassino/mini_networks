"""Mini DINO: self-distillation with no labels, on a tiny ViT.

DINO (Caron et al., 2021) learns visual representations without labels by
self-distillation: a *student* network is trained to match the output
distribution of a *teacher* network on a different augmented view of the same
image. The teacher is not trained by gradients at all — it is an exponential
moving average (EMA) of the student, so the student is forever chasing a
slightly better, smoothed version of itself:

    teacher_params ← m · teacher_params + (1 − m) · student_params

Collapse (everyone outputs the same distribution) is prevented by two
asymmetries applied only to the teacher: *centering* (subtract a running mean
of teacher outputs, so no single prototype dominates) and *sharpening* (a much
lower softmax temperature than the student, so the teacher commits to
confident targets). The loss is a plain cross-entropy between the sharpened,
centered teacher distribution and the student's log-softmax, computed
cross-view (student sees view 1 against teacher's view 2 and vice versa), with
a stop-gradient through the teacher.

This implementation:

    [B,1,28,28] x 2 views (contrastive MNIST augmentations)
      student:  MiniViT features (d=64) -> MLP head 64->128->64 -> K=64 logits
      teacher:  same architecture, EMA of the student, no_grad
      loss:     CE( softmax((t − center)/0.04), log_softmax(s/0.1) ), cross-view

Deliberately simplified vs the paper: two global views only (no multi-crop),
K = 64 prototypes instead of 65536, a plain linear prototype layer instead of
weight-normalised, MNIST instead of ImageNet, and the backbone is the same
tiny MiniViT used by the supervised `vit` model (reused via
``forward_features`` so the two stay comparable).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_networks.models.vit.model import MiniViT


class DINOHead(nn.Module):
    """Projection MLP -> L2-normalise -> prototype logits."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim),
        )
        self.prototypes = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.prototypes(x)


def _make_branch(
    patch_size: int, d_model: int, n_heads: int, n_layers: int, mlp_dim: int,
    proj_hidden: int, out_dim: int,
) -> tuple[MiniViT, DINOHead]:
    backbone = MiniViT(
        patch_size=patch_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, mlp_dim=mlp_dim,
    )
    return backbone, DINOHead(d_model, proj_hidden, out_dim)


class MiniDINO(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        mlp_dim: int = 128,
        proj_hidden: int = 128,
        out_dim: int = 64,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        ema_decay: float = 0.996,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.ema_decay = ema_decay
        self.center_momentum = center_momentum

        args = (patch_size, d_model, n_heads, n_layers, mlp_dim, proj_hidden, out_dim)
        self.student_backbone, self.student_head = _make_branch(*args)
        self.teacher_backbone, self.teacher_head = _make_branch(*args)
        # Teacher starts as a copy of the student and only ever moves by EMA.
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self._teacher_params():
            p.requires_grad_(False)

        self.register_buffer("center", torch.zeros(1, out_dim))

    def _student_params(self):
        yield from self.student_backbone.parameters()
        yield from self.student_head.parameters()

    def _teacher_params(self):
        yield from self.teacher_backbone.parameters()
        yield from self.teacher_head.parameters()

    def forward_views(self, v1: torch.Tensor, v2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Student and teacher prototype logits, cross-view aligned row-by-row.

        Returns (student [2B,K], teacher [2B,K]) where row i of the teacher is
        the OTHER view of row i of the student — so the loss can zip them.
        """
        s1 = self.student_head(self.student_backbone.forward_features(v1))
        s2 = self.student_head(self.student_backbone.forward_features(v2))
        with torch.no_grad():
            t1 = self.teacher_head(self.teacher_backbone.forward_features(v1))
            t2 = self.teacher_head(self.teacher_backbone.forward_features(v2))
        return torch.cat([s1, s2], dim=0), torch.cat([t2, t1], dim=0)

    def dino_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        targets = F.softmax((teacher_logits - self.center) / self.teacher_temp, dim=-1).detach()
        log_probs = F.log_softmax(student_logits / self.student_temp, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1).mean()
        if self.training:
            self._update_center(teacher_logits)
        return loss

    @torch.no_grad()
    def _update_center(self, teacher_logits: torch.Tensor) -> None:
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)

    @torch.no_grad()
    def update_teacher(self) -> None:
        for ps, pt in zip(self._student_params(), self._teacher_params()):
            pt.mul_(self.ema_decay).add_(ps.detach(), alpha=1 - self.ema_decay)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Downstream representation: the teacher backbone's CLS features."""
        return self.teacher_backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)
