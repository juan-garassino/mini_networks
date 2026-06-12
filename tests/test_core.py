"""Tests for core: config, data registry (all 4 MNIST modes), logger."""
import os
import tempfile

import pytest
import torch

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader, get_dataset
from mini_networks.core.logging.logger import Logger


DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


class TestConfig:
    def test_defaults(self):
        c = BaseConfig()
        assert c.batch_size == 32
        assert c.fast_demo is False

    def test_fast_demo_reduces_epochs(self):
        c = BaseConfig(epochs=20, fast_demo=True)
        assert c.effective_epochs == 1

    def test_fast_demo_caps_batch_size(self):
        c = BaseConfig(batch_size=64, fast_demo=True)
        assert c.effective_batch_size <= 16


class TestLogger:
    def test_log_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test-run")
            logger.log_metric(0, "loss", 1.23)
            logger.log_metric(1, "loss", 0.99)
            metrics = logger.read_metrics()
            assert len(metrics) == 2
            assert metrics[0]["key"] == "loss"
            assert metrics[0]["value"] == pytest.approx(1.23)

    def test_log_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test-run")
            logger.log_config({"model": "test", "lr": 0.001})
            assert (logger.run_dir / "config.yaml").exists()

    def test_artifact_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test-run")
            p = logger.artifact_path("model.pt")
            assert "artifacts" in str(p)


class TestDataRegistryClassification:
    def test_mnist_classification_length(self):
        ds = get_dataset("mnist", DATA_ROOT, split="train", task="classification", fast_demo=True)
        assert len(ds) <= 512

    def test_mnist_classification_item_shape(self):
        ds = get_dataset("mnist", DATA_ROOT, split="train", task="classification", fast_demo=True)
        img, label = ds[0]
        assert img.shape == (1, 28, 28)
        assert isinstance(int(label), int)

    def test_dataloader_classification(self):
        dl = get_dataloader("mnist", DATA_ROOT, split="train", task="classification", batch_size=8, fast_demo=True)
        images, labels = next(iter(dl))
        assert images.shape[1:] == (1, 28, 28)
        assert labels.shape[0] == images.shape[0]


class TestDataRegistryBinarySegmentation:
    def test_binary_seg_mask_shape(self):
        ds = get_dataset("mnist", DATA_ROOT, split="train", task="binary_segmentation", fast_demo=True)
        img, mask = ds[0]
        assert img.shape == (1, 28, 28)
        assert mask.shape == (28, 28)

    def test_binary_seg_mask_values(self):
        ds = get_dataset("mnist", DATA_ROOT, split="train", task="binary_segmentation", fast_demo=True)
        _, mask = ds[0]
        unique = torch.unique(mask)
        assert set(unique.tolist()).issubset({0, 1})


class TestDataRegistryMulticlassSegmentation:
    def test_multiclass_seg_shape(self):
        ds = get_dataset("mnist", DATA_ROOT, split="train", task="multiclass_segmentation", fast_demo=True)
        img, mask = ds[0]
        assert img.shape == (1, 28, 28)
        assert mask.shape == (28, 28)

    def test_multiclass_seg_classes(self):
        ds = get_dataset("mnist", DATA_ROOT, split="train", task="multiclass_segmentation", fast_demo=True)
        _, mask = ds[0]
        assert mask.min() >= 0
        assert mask.max() <= 11


class TestDataRegistryDetection:
    def test_detection_canvas_shape(self):
        ds = get_dataset("mnist", DATA_ROOT, split="train", task="detection", fast_demo=True, canvas_size=56)
        canvas, label, bbox = ds[0]
        assert canvas.shape == (1, 56, 56)

    def test_detection_bbox_normalized(self):
        ds = get_dataset("mnist", DATA_ROOT, split="train", task="detection", fast_demo=True)
        _, _, bbox = ds[0]
        assert bbox.shape == (4,)
        assert bbox.min() >= 0.0
        assert bbox.max() <= 1.0
