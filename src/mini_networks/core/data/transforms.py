"""Shared transforms for all data modes."""
from __future__ import annotations

import numpy as np
import torch


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    """Normalize to [-1, 1] from [0, 1]."""
    return x * 2.0 - 1.0


def make_binary_mask(image: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Create binary mask: 1 where pixel > threshold, else 0."""
    return (image > threshold).long().squeeze(0)


def make_composite_image(img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
    """Overlay two grayscale images by taking the element-wise maximum."""
    return torch.maximum(img_a, img_b)


def make_multiclass_mask(
    img_a: torch.Tensor,
    label_a: int,
    img_b: torch.Tensor,
    label_b: int,
    num_classes: int = 12,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Build a 12-class segmentation mask from two overlaid MNIST digits.
    Classes 0-9: digit pixels (exclusive), 10: background, 11: intersection.
    Returns LongTensor of shape (H, W).
    """
    h, w = img_a.shape[-2], img_a.shape[-1]
    mask_a = (img_a.squeeze(0) > threshold)
    mask_b = (img_b.squeeze(0) > threshold)
    intersection = mask_a & mask_b
    only_a = mask_a & ~intersection
    only_b = mask_b & ~intersection
    background = ~mask_a & ~mask_b

    result = torch.zeros(h, w, dtype=torch.long)
    result[background] = 10
    result[intersection] = 11
    result[only_a] = label_a
    result[only_b] = label_b
    return result


def place_on_canvas(
    image: torch.Tensor,
    canvas_size: int = 56,
) -> tuple[torch.Tensor, list[int]]:
    """
    Place a 28x28 digit randomly on a canvas_size x canvas_size canvas.
    Returns (canvas_tensor [1, H, W], bbox [x1, y1, x2, y2]).
    """
    _, h, w = image.shape
    max_row = canvas_size - h
    max_col = canvas_size - w
    row = np.random.randint(0, max_row + 1)
    col = np.random.randint(0, max_col + 1)
    canvas = torch.zeros(1, canvas_size, canvas_size, dtype=image.dtype)
    canvas[:, row:row + h, col:col + w] = image
    bbox = [col, row, col + w, row + h]  # x1, y1, x2, y2
    return canvas, bbox
