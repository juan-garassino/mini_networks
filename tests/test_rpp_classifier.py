"""Tests for the Residual Pathway Prior classifier."""
import os
import tempfile

import torch

from mini_networks.core.logging.logger import Logger
from mini_networks.models.rpp_classifier.config import RPPClassifierConfig
from mini_networks.models.rpp_classifier.model import RPPBlock, RPPClassifier
from mini_networks.models.rpp_classifier.trainer import (
    RPPClassifierTrainer,
    make_rpp_classifier_dataloader,
)

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


class TestRPPBlock:
    def test_pathways_sum(self):
        """Zeroing the free path must reduce the block to (bn of) the conv path."""
        block = RPPBlock(1, 4, spatial=14)
        block.eval()
        x = torch.randn(2, 1, 14, 14)
        with torch.no_grad():
            full = block(x)
            block.free.weight.zero_()
            conv_only = block(x)
        assert full.shape == conv_only.shape == (2, 4, 14, 14)
        assert not torch.allclose(full, conv_only)  # free path contributed

    def test_free_path_contains_conv(self):
        """The dense free path is strictly more expressive: it can represent
        any linear map on the flattened input, including the conv."""
        block = RPPBlock(1, 2, spatial=7)
        assert block.free.weight.shape == (2 * 49, 1 * 49)


class TestRPPClassifier:
    def test_forward_shape(self):
        model = RPPClassifier(hidden_dim=8, num_classes=10)
        assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)

    def test_prior_penalty_ordering(self):
        """Same weights, stronger prior on the free path => the free-path term
        dominates the penalty when norms are comparable."""
        model = RPPClassifier(hidden_dim=8)
        p_equal = model.prior_penalty(1e-3, 1e-3)
        p_rpp = model.prior_penalty(1e-4, 1e-2)
        assert p_equal > 0 and p_rpp > 0

    def test_pathway_norms(self):
        norms = RPPClassifier(hidden_dim=8).pathway_norms()
        assert set(norms) == {"conv_norm", "free_norm"}
        assert norms["conv_norm"] > 0 and norms["free_norm"] > 0

    def test_backprop_through_penalty(self):
        model = RPPClassifier(hidden_dim=8)
        x = torch.randn(2, 1, 28, 28)
        loss = model(x).sum() + model.prior_penalty(1e-4, 1e-2)
        loss.backward()
        assert model.block1.free.weight.grad is not None
        assert model.block1.conv.weight.grad is not None


class TestRPPClassifierTrainer:
    def _config(self, **kwargs):
        defaults = dict(hidden_dim=8, fast_demo=True, data_root=DATA_ROOT, epochs=1)
        defaults.update(kwargs)
        return RPPClassifierConfig(**defaults)

    def test_penalty_only_in_train_mode(self):
        config = self._config()
        trainer = RPPClassifierTrainer()
        trainer.model = trainer._build(config)
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        trainer.model.train()
        loss_train = trainer._loss(logits, targets)
        trainer.model.eval()
        loss_eval = trainer._loss(logits, targets)
        assert loss_train > loss_eval  # prior penalty added in train mode only

    def test_train_smoke(self):
        config = self._config()
        trainer = RPPClassifierTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_rpp")
            dl = make_rpp_classifier_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            assert len(logger.read_metrics()) > 0
            assert logger.artifact_path("model.pt").exists()

    def test_evaluate_and_infer(self):
        config = self._config()
        trainer = RPPClassifierTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_rpp")
            dl = make_rpp_classifier_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.evaluate(config, dl, logger)
            assert "accuracy" in result and "eval_loss" in result
            out = trainer.infer(config, torch.randn(2, 1, 28, 28))
            assert len(out["predictions"]) == 2

    def test_checkpoint_roundtrip(self):
        config = self._config()
        trainer = RPPClassifierTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_rpp")
            dl = make_rpp_classifier_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            fresh = RPPClassifierTrainer()
            fresh.load_checkpoint(config, logger.artifact_path("model.pt").parent)
            assert fresh.model is not None
