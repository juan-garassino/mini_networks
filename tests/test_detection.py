"""Detection: canvas generation, bbox, dual-head forward, trainer 1-step."""
import os
import tempfile

import torch

from mini_networks.models.detection.config import DetectionConfig
from mini_networks.models.detection.model import DigitDetector, detection_loss
from mini_networks.models.detection.trainer import DetectionTrainer, make_detection_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.core.data.transforms import place_on_canvas

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


class TestPlaceOnCanvas:
    def test_canvas_shape(self):
        img = torch.rand(1, 28, 28)
        canvas, bbox = place_on_canvas(img, canvas_size=56)
        assert canvas.shape == (1, 56, 56)

    def test_bbox_valid(self):
        img = torch.rand(1, 28, 28)
        canvas, bbox = place_on_canvas(img, canvas_size=56)
        x1, y1, x2, y2 = bbox
        assert 0 <= x1 <= x2 <= 56
        assert 0 <= y1 <= y2 <= 56
        assert x2 - x1 == 28
        assert y2 - y1 == 28


class TestDigitDetector:
    def test_forward_shape(self):
        model = DigitDetector(num_classes=10)
        x = torch.randn(2, 1, 56, 56)
        cls_logits, bbox_pred = model(x)
        assert cls_logits.shape == (2, 10)
        assert bbox_pred.shape == (2, 4)

    def test_bbox_output_range(self):
        model = DigitDetector(num_classes=10)
        x = torch.randn(2, 1, 56, 56)
        _, bbox_pred = model(x)
        assert bbox_pred.min() >= 0.0
        assert bbox_pred.max() <= 1.0

    def test_detection_loss(self):
        cls_logits = torch.randn(4, 10)
        bbox_pred = torch.sigmoid(torch.randn(4, 4))
        labels = torch.randint(0, 10, (4,))
        bboxes = torch.rand(4, 4)
        loss = detection_loss(cls_logits, bbox_pred, labels, bboxes)
        assert loss.item() > 0


class TestDetectionTrainer:
    def test_train_smoke(self):
        config = DetectionConfig(
            canvas_size=56, num_classes=10,
            fast_demo=True, data_root=DATA_ROOT, epochs=1,
        )
        trainer = DetectionTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test")
            dl = make_detection_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            assert len(metrics) > 0

    def test_evaluate(self):
        config = DetectionConfig(
            canvas_size=56, num_classes=10,
            fast_demo=True, data_root=DATA_ROOT, epochs=1,
        )
        trainer = DetectionTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test")
            dl = make_detection_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.evaluate(config, dl, logger)
            assert "eval_accuracy" in result
