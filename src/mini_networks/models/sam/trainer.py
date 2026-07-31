"""Mini-SAM trainer: on-the-fly prompt sampling + promptability evidence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.sam.config import SAMConfig
from mini_networks.models.sam.model import MiniSAM, dice_loss

import logging

log = logging.getLogger(__name__)


def _random_fg_point(mask: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Random foreground (y, x) of a [H, W] mask, normalized to [0, 1]."""
    fg = mask.nonzero()
    idx = int(torch.randint(0, len(fg), (1,), generator=g)) if len(fg) else 0
    p = fg[idx].float() if len(fg) else torch.tensor([mask.shape[0] / 2] * 2)
    return p / mask.shape[-1]


def interior_click(mask: torch.Tensor) -> torch.Tensor:
    """Deterministic eval prompt: center of mass snapped to the nearest
    foreground pixel (the raw COM lands in the hole of 0/4/6/8/9)."""
    fg = mask.nonzero().float()
    if not len(fg):
        return torch.tensor([0.5, 0.5])
    com = fg.mean(dim=0)
    nearest = fg[(fg - com).pow(2).sum(dim=1).argmin()]
    return nearest / mask.shape[-1]


def _bbox_corners(mask: torch.Tensor) -> torch.Tensor:
    fg = mask.nonzero().float()
    if not len(fg):
        return torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    return torch.stack([fg.min(dim=0).values, fg.max(dim=0).values]) / mask.shape[-1]


class SAMTrainer(BaseTrainer):
    def __init__(self):
        self.model: MiniSAM | None = None

    def _build(self, config: SAMConfig) -> MiniSAM:
        return MiniSAM(
            embed_dim=config.embed_dim,
            n_heads=config.n_heads,
            n_decoder_layers=config.n_decoder_layers,
            n_masks=config.n_masks,
        ).to(config.device)

    def _sample_prompts(self, mask_t: torch.Tensor, mask_o: torch.Tensor,
                        config: SAMConfig, g: torch.Generator):
        """One target mask -> (coords [P,2], types [P]); may add a negative
        click on the other digit or use a box instead of a point."""
        if torch.rand(1, generator=g).item() < config.box_prompt_prob:
            return _bbox_corners(mask_t), torch.tensor([2, 3])
        coords = [_random_fg_point(mask_t, g)]
        types = [0]
        if torch.rand(1, generator=g).item() < config.neg_prompt_prob:
            coords.append(_random_fg_point(mask_o, g))
            types.append(1)
        return torch.stack(coords), torch.tensor(types)

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, SAMConfig)
        model = self._build(config)
        self.model = model
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        logger.log_config(config.model_dump())
        g = torch.Generator().manual_seed(config.seed)

        for epoch in range(config.effective_epochs):
            model.train()
            total = 0.0
            wins = torch.zeros(config.n_masks)
            for image, mask_a, mask_b in dataloader:
                B = image.shape[0]
                coords_list, types_list, targets = [], [], []
                for i in range(B):
                    # the click chooses the target digit — promptability is the task
                    pick_a = torch.rand(1, generator=g).item() < 0.5
                    t, o = (mask_a[i], mask_b[i]) if pick_a else (mask_b[i], mask_a[i])
                    c, ty = self._sample_prompts(t, o, config, g)
                    coords_list.append(c)
                    types_list.append(ty)
                    targets.append(t)
                P = max(c.shape[0] for c in coords_list)
                coords = torch.zeros(B, P, 2)
                types = torch.zeros(B, P, dtype=torch.long)
                for i, (c, ty) in enumerate(zip(coords_list, types_list)):
                    coords[i, : c.shape[0]] = c
                    types[i, : ty.shape[0]] = ty
                target = torch.stack(targets).to(config.device)
                image = image.to(config.device)

                masks, iou_pred = model(image, coords.to(config.device),
                                        types.to(config.device))
                probs = torch.sigmoid(masks)
                bce = F.binary_cross_entropy_with_logits(
                    masks, target.unsqueeze(1).expand_as(masks), reduction="none"
                ).mean(dim=(-2, -1))                                    # [B, n_masks]
                dic = torch.stack(
                    [dice_loss(probs[:, h], target) for h in range(config.n_masks)],
                    dim=1,
                )
                per_head = bce + dic
                best, best_idx = per_head.min(dim=1)  # PER-SAMPLE min (not per-batch)
                wins += torch.bincount(best_idx.cpu(), minlength=config.n_masks).float()

                with torch.no_grad():  # IoU targets from detached, thresholded preds
                    hard = (probs > 0.5).float()
                    inter = (hard * target.unsqueeze(1)).sum(dim=(-2, -1))
                    union = ((hard + target.unsqueeze(1)) > 0).float().sum(dim=(-2, -1))
                    actual_iou = inter / (union + 1e-6)                 # [B, n_masks]
                    winner_iou = actual_iou.gather(1, best_idx.unsqueeze(1)).squeeze(1)
                iou_loss = F.mse_loss(
                    iou_pred.gather(1, best_idx.unsqueeze(1)).squeeze(1), winner_iou
                )
                loss = best.mean() + iou_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            metrics = {"loss": total / max(1, len(dataloader)), "epoch": epoch}
            metrics.update({f"head{h}_wins": wins[h].item() for h in range(config.n_masks)})
            logger.log_metrics(epoch, metrics)
            log.info(f"  epoch {epoch}  loss {metrics['loss']:.4f}  wins {wins.tolist()}")

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))

    @torch.no_grad()
    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        """Gate metric = IoU on the CLICKED digit; wrong_prompt_iou = the same
        prediction scored against the OTHER digit — the gap is the evidence
        that the prompt is actually steering the mask."""
        assert isinstance(config, SAMConfig)
        if self.model is None:
            self.model = self._build(config)
        model = self.model
        model.eval()
        iou_sum = wrong_sum = n = 0.0
        for image, mask_a, mask_b in dataloader:
            coords = torch.stack([interior_click(m) for m in mask_a]).unsqueeze(1)
            types = torch.zeros(image.shape[0], 1, dtype=torch.long)
            masks, iou_pred = model(image.to(config.device),
                                    coords.to(config.device), types.to(config.device))
            best = iou_pred.argmax(dim=1)  # SAM inference: self-rated best mask
            pred = (torch.sigmoid(
                masks.gather(1, best.view(-1, 1, 1, 1).expand(-1, 1, *masks.shape[-2:]))
            ).squeeze(1) > 0.5).float().cpu()
            for m, gt, other in zip(pred, mask_a, mask_b):
                inter = (m * gt).sum()
                union = ((m + gt) > 0).float().sum()
                iou_sum += (inter / (union + 1e-6)).item()
                inter_o = (m * other).sum()
                union_o = ((m + other) > 0).float().sum()
                wrong_sum += (inter_o / (union_o + 1e-6)).item()
                n += 1
        return {
            "eval_iou": iou_sum / max(1, n),
            "wrong_prompt_iou": wrong_sum / max(1, n),
        }

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, SAMConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if not isinstance(inputs, dict) or "images" not in inputs:
            raise ValueError("infer() expects {'images': [B,1,56,56], 'points': [[y,x]], 'labels': [...]}")
        image = inputs["images"].to(config.device)
        pts = torch.tensor(inputs.get("points", [[28, 28]]), dtype=torch.float32)
        coords = (pts / image.shape[-1]).unsqueeze(0).expand(image.shape[0], -1, -1)
        types = torch.tensor(inputs.get("labels", [1]), dtype=torch.long)
        types = (1 - types).unsqueeze(0).expand(image.shape[0], -1)  # label 1 = positive = type 0
        self.model.eval()
        with torch.no_grad():
            masks, iou_pred = self.model(image, coords.to(config.device),
                                         types.to(config.device))
        return {
            "masks": torch.sigmoid(masks).cpu(),
            "iou_pred": iou_pred.cpu().tolist(),
        }

    def load_checkpoint(self, config: BaseConfig, artifacts_dir) -> None:
        assert isinstance(config, SAMConfig)
        state = torch.load(
            Path(artifacts_dir) / "model.pt", map_location=config.device, weights_only=True
        )
        self.model = self._build(config)
        self.model.load_state_dict(state)
        self.model.eval()


def make_sam_dataloader(config: SAMConfig, split: str = "train") -> DataLoader:
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        batch_size=config.effective_batch_size,
        fast_demo=config.effective_fast_demo,
        sample_limit=config.dataset_sample_limit,
        seed=config.seed,
    )
