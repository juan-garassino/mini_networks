"""Tests for the double-descent width-sweep composition."""
import os
import tempfile

import torch

from mini_networks.compositions.double_descent import (
    DoubleDescent,
    DoubleDescentConfig,
    _LabelNoise,
    _render_curve,
    _WidthMLP,
)
from mini_networks.core.logging.logger import Logger

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


class _FakeDataset:
    def __init__(self, n=100):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.zeros(1, 28, 28), idx % 10


class TestLabelNoise:
    def test_flip_fraction_and_determinism(self):
        base = _FakeDataset(200)
        a = _LabelNoise(base, n_take=100, noise=0.2, num_classes=10, seed=7)
        b = _LabelNoise(base, n_take=100, noise=0.2, num_classes=10, seed=7)
        assert a.n_flipped == b.n_flipped
        assert [y for _, y in a._items] == [y for _, y in b._items]
        assert 5 <= a.n_flipped <= 40  # ~20 of 100, generous band

    def test_no_noise_keeps_labels(self):
        base = _FakeDataset(50)
        ds = _LabelNoise(base, n_take=50, noise=0.0, num_classes=10)
        assert ds.n_flipped == 0


class TestRenderCurve:
    def test_writes_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "curve.png")
            _render_curve([2, 8, 32, 128], [0.4, 0.2, 0.3, 0.15], path)
            assert os.path.exists(path)


class TestWidthMLP:
    def test_forward(self):
        model = _WidthMLP(width=4)
        assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)


class TestDoubleDescentSmoke:
    def test_train_all_s_tier(self):
        config = DoubleDescentConfig(fast_demo=True, data_root=DATA_ROOT)
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_dd")
            result = DoubleDescent().train_all(config, logger)
            assert result["widths"] == [2, 4]  # S tier sweeps the first two widths
            assert len(result["test_errors"]) == 2
            assert all(0.0 <= e <= 1.0 for e in result["test_errors"])
            assert logger.artifact_path("double_descent.png").exists()
            keys = {m.get("key") for m in logger.read_metrics()}
            assert {"loss", "test_accuracy", "train_accuracy"} <= keys
