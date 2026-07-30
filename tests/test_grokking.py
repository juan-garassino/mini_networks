"""Tests for grokking: dataset split hygiene, model, step-based trainer."""
import tempfile

import torch

from mini_networks.core.data.registry import ModularArithmeticDataset
from mini_networks.core.logging.logger import Logger
from mini_networks.models.grokking.config import GrokkingConfig
from mini_networks.models.grokking.model import GrokkingTransformer
from mini_networks.models.grokking.trainer import GrokkingTrainer, make_grokking_dataloader


class TestModularArithmeticDataset:
    def test_split_disjoint_and_exhaustive(self):
        """Train/val pairs are disjoint and together cover every valid pair —
        leakage here would fake the grokking curve entirely."""
        p = 23
        train = ModularArithmeticDataset(p=p, split="train")
        val = ModularArithmeticDataset(p=p, split="val")
        train_pairs = set(train._pairs)
        val_pairs = set(val._pairs)
        assert not train_pairs & val_pairs
        assert len(train_pairs) + len(val_pairs) == p * (p - 1)

    def test_deterministic(self):
        a = ModularArithmeticDataset(p=23, split="train")
        b = ModularArithmeticDataset(p=23, split="train")
        assert a._pairs == b._pairs

    def test_division_correct(self):
        ds = ModularArithmeticDataset(p=23, split="train")
        tokens, answer = ds[0]
        a, op, b, eq = tokens.tolist()
        assert op == 23 and eq == 24
        assert (answer * b) % 23 == a  # (a/b)*b == a mod p

    def test_fast_demo_caps(self):
        ds = ModularArithmeticDataset(p=97, fast_demo=True, split="train")
        assert len(ds) == 256


class TestGrokkingTransformer:
    def test_forward_shape(self):
        model = GrokkingTransformer(p=23, d_model=32, n_heads=2, n_layers=2)
        tokens = torch.tensor([[1, 23, 2, 24], [5, 23, 7, 24]])
        assert model(tokens).shape == (2, 23)

    def test_backprop(self):
        model = GrokkingTransformer(p=23, d_model=32, n_heads=2, n_layers=2)
        tokens = torch.tensor([[1, 23, 2, 24]])
        model(tokens).sum().backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


class TestGrokkingTrainer:
    def _config(self, **kwargs):
        defaults = dict(
            p=23, d_model=32, n_heads=2, n_layers=2,
            fast_demo=True, epochs=1,
        )
        defaults.update(kwargs)
        return GrokkingConfig(**defaults)

    def test_step_cap_honors_tier(self):
        config = self._config()  # fast_demo -> S tier
        assert config.limit_steps(config.n_train_steps, s_cap=50, m_cap=20_000) == 50

    def test_train_smoke_and_metrics(self):
        config = self._config()
        trainer = GrokkingTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_grokking")
            dl = make_grokking_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            assert len(metrics) > 0
            keys = {m.get("key") for m in metrics}
            assert {"loss", "train_accuracy", "val_accuracy"} <= keys
            assert logger.artifact_path("model.pt").exists()

    def test_evaluate_and_infer(self):
        config = self._config()
        trainer = GrokkingTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_grokking")
            dl = make_grokking_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.evaluate(config, dl, logger)
            assert 0.0 <= result["accuracy"] <= 1.0
            out = trainer.infer(config, {"a": 12, "b": 5})
            assert 0 <= out["prediction"] < 23
            assert (out["expected"] * 5) % 23 == 12

    def test_checkpoint_roundtrip(self):
        config = self._config()
        trainer = GrokkingTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_grokking")
            dl = make_grokking_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            fresh = GrokkingTrainer()
            fresh.load_checkpoint(config, logger.artifact_path("model.pt").parent)
            assert fresh.model is not None
