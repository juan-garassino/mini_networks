"""Multitask vision head: classification + segmentation + detection."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import (
    _load_torchvision_dataset,
    make_binary_mask,
    place_on_canvas,
)
from mini_networks.core.logging.logger import Logger


class MultiTaskVisionConfig(BaseConfig):
    model_name: str = "multitask_vision"
    base_channels: int = 32
    num_classes: int = 10
    canvas_size: int = 56
    dataset: str = "mnist"


class MultiTaskDataset(Dataset):
    """Returns (canvas_image, class_label, seg_mask, bbox)."""

    def __init__(
        self,
        data_root: str,
        train: bool,
        canvas_size: int,
        fast_demo: bool,
        dataset: str,
        sample_limit: int | None = None,
    ):
        import torchvision
        import torchvision.transforms as T

        ds_cls = torchvision.datasets.MNIST if dataset == "mnist" else torchvision.datasets.FashionMNIST
        dataset_name = "MNIST" if dataset == "mnist" else "FashionMNIST"
        ds = _load_torchvision_dataset(
            ds_cls,
            data_root=data_root,
            train=train,
            transform=T.ToTensor(),
            name=dataset_name,
        )
        self._data = ds
        self.canvas_size = canvas_size
        base_limit = 256 if fast_demo else len(ds)
        self._limit = min(base_limit, sample_limit) if sample_limit is not None else base_limit

    def __len__(self) -> int:
        return self._limit

    def __getitem__(self, idx: int):
        image, label = self._data[idx]
        canvas, bbox = place_on_canvas(image, self.canvas_size)
        mask = make_binary_mask(canvas, threshold=0.0)
        bbox_norm = torch.tensor([c / self.canvas_size for c in bbox], dtype=torch.float32)
        return canvas, int(label), mask, bbox_norm


class MultiTaskModel(nn.Module):
    def __init__(self, base_channels: int = 32, num_classes: int = 10):
        super().__init__()
        c = base_channels
        self.backbone = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c, c * 2, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c * 2, num_classes),
        )
        self.seg_head = nn.Sequential(
            nn.Conv2d(c * 2, c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c, 1, 1),
            nn.Sigmoid(),
        )
        self.det_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        logits = self.cls_head(feat)
        seg = self.seg_head(feat)
        bbox = self.det_head(feat)
        return logits, seg, bbox


class MultiTaskVision:
    def __init__(self):
        self.model: MultiTaskModel | None = None

    def _build(self, config: MultiTaskVisionConfig) -> MultiTaskModel:
        return MultiTaskModel(base_channels=config.base_channels, num_classes=config.num_classes).to(config.device)

    def train(self, config: MultiTaskVisionConfig, logger: Logger) -> None:
        ds = MultiTaskDataset(
            data_root=config.data_root,
            train=True,
            canvas_size=config.canvas_size,
            fast_demo=config.effective_fast_demo,
            dataset=config.dataset,
            sample_limit=config.dataset_sample_limit,
        )
        dl = DataLoader(ds, batch_size=config.effective_batch_size, shuffle=True, num_workers=0)
        model = self._build(config)
        self.model = model
        opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        for epoch in range(config.effective_epochs):
            model.train()
            total = 0.0
            for images, labels, masks, bboxes in dl:
                images = images.to(config.device)
                labels = labels.to(config.device)
                masks = masks.to(config.device)
                bboxes = bboxes.to(config.device)
                logits, seg, bbox = model(images)
                seg = F.interpolate(
                    seg,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                cls_loss = F.cross_entropy(logits, labels)
                seg_loss = F.binary_cross_entropy(seg.squeeze(1), masks.float())
                det_loss = F.mse_loss(bbox, bboxes)
                loss = cls_loss + seg_loss + det_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            logger.log_metrics(epoch, {"loss": total / max(1, len(dl))})

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))
