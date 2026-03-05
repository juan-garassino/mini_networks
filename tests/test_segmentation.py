"""Segmentation: binary + multiclass UNet forward + composite mask generation."""
import os
import tempfile

import torch

from mini_networks.models.segmentation.config import SegmentationConfig
from mini_networks.models.segmentation.unet import SegUNet, dice_loss, multiclass_dice_loss
from mini_networks.models.segmentation.trainer import SegmentationTrainer, make_segmentation_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.core.data.transforms import make_multiclass_mask, make_composite_image

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


class TestSegUNet:
    def test_binary_forward_shape(self):
        model = SegUNet(in_channels=1, out_channels=1, base_channels=8)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 1, 28, 28)

    def test_binary_output_range(self):
        model = SegUNet(in_channels=1, out_channels=1, base_channels=8)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_multiclass_forward_shape(self):
        model = SegUNet(in_channels=1, out_channels=12, base_channels=8)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 12, 28, 28)

    def test_dice_loss(self):
        pred = torch.sigmoid(torch.randn(4, 28, 28))
        target = torch.randint(0, 2, (4, 28, 28))
        loss = dice_loss(pred, target)
        assert 0.0 <= loss.item() <= 1.0


class TestCompositeMask:
    def test_multiclass_mask_shape(self):
        img_a = torch.rand(1, 28, 28)
        img_b = torch.rand(1, 28, 28)
        mask = make_multiclass_mask(img_a, 3, img_b, 7)
        assert mask.shape == (28, 28)

    def test_multiclass_mask_classes(self):
        img_a = torch.rand(1, 28, 28)
        img_b = torch.rand(1, 28, 28)
        mask = make_multiclass_mask(img_a, 3, img_b, 7)
        assert mask.min() >= 0
        assert mask.max() <= 11

    def test_composite_image(self):
        img_a = torch.rand(1, 28, 28)
        img_b = torch.rand(1, 28, 28)
        composite = make_composite_image(img_a, img_b)
        assert composite.shape == (1, 28, 28)


class TestSegmentationTrainer:
    def test_binary_train_smoke(self):
        config = SegmentationConfig(
            task_mode="binary", num_classes=1, base_channels=8,
            fast_demo=True, data_root=DATA_ROOT, epochs=1,
        )
        trainer = SegmentationTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test")
            dl = make_segmentation_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            assert len(metrics) > 0

    def test_multiclass_train_smoke(self):
        config = SegmentationConfig(
            task_mode="multiclass", num_classes=12, base_channels=8,
            fast_demo=True, data_root=DATA_ROOT, epochs=1,
        )
        trainer = SegmentationTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test")
            dl = make_segmentation_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            assert len(metrics) > 0
