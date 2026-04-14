"""Segment-then-detect composition: segment digit, then derive bbox."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.models.segmentation.unet import SegUNet, dice_loss


class SegmentThenDetectConfig(BaseConfig):
    model_name: str = "segment_then_detect"
    base_channels: int = 32
    dataset: str = "mnist"


def mask_to_bbox(mask: torch.Tensor) -> torch.Tensor:
    """mask: [B, H, W] -> bbox [B, 4] in (x1,y1,x2,y2) normalized."""
    bboxes = []
    for m in mask:
        ys, xs = torch.where(m > 0)
        if len(xs) == 0:
            bboxes.append(torch.tensor([0, 0, 0, 0], dtype=torch.float32))
            continue
        x1, x2 = xs.min().item(), xs.max().item()
        y1, y2 = ys.min().item(), ys.max().item()
        h, w = m.shape
        bboxes.append(torch.tensor([x1 / w, y1 / h, x2 / w, y2 / h], dtype=torch.float32))
    return torch.stack(bboxes, dim=0)


class SegmentThenDetect:
    def __init__(self):
        self.model: SegUNet | None = None

    def _build(self, config: SegmentThenDetectConfig) -> SegUNet:
        return SegUNet(in_channels=1, out_channels=1, base_channels=config.base_channels).to(config.device)

    def train(self, config: SegmentThenDetectConfig, logger: Logger) -> None:
        dl = get_dataloader(
            name=config.dataset,
            data_root=config.data_root,
            split="train",
            task="binary_segmentation",
            batch_size=config.effective_batch_size,
            fast_demo=config.effective_fast_demo,
            sample_limit=config.dataset_sample_limit,
        )
        model = self._build(config)
        self.model = model
        opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        for epoch in range(config.effective_epochs):
            model.train()
            total = 0.0
            for images, masks in dl:
                images, masks = images.to(config.device), masks.to(config.device)
                preds = model(images)
                loss = F.binary_cross_entropy(preds.squeeze(1), masks.float()) + dice_loss(
                    preds.squeeze(1), masks
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            logger.log_metrics(epoch, {"seg_loss": total / max(1, len(dl))})

        torch.save(model.state_dict(), logger.artifact_path("seg_unet.pt"))

    @torch.no_grad()
    def infer_bbox(self, config: SegmentThenDetectConfig, images: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Train first.")
        model = self.model
        model.eval()
        images = images.to(config.device)
        preds = model(images).squeeze(1)
        masks = (preds > 0.5).float()
        return mask_to_bbox(masks.cpu())
