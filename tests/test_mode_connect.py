"""Tests for the mode-connectivity simplex composition."""
import os
import tempfile

import torch

from mini_networks.compositions.mode_connect import ModeConnect, ModeConnectConfig
from mini_networks.core.logging.logger import Logger
from mini_networks.models.classifier.model import SmallCNN

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


class TestMerge:
    def test_exact_at_vertices(self):
        """Weight (1,0,0) must reproduce vertex 0 exactly."""
        m1 = SmallCNN(8).state_dict()
        m2 = SmallCNN(8).state_dict()
        m3 = SmallCNN(8).state_dict()
        merged = ModeConnect._merge([m1, m2, m3], torch.tensor([1.0, 0.0, 0.0]))
        for k in m1:
            if m1[k].is_floating_point():
                assert torch.allclose(merged[k], m1[k])

    def test_midpoint(self):
        m1 = {"w": torch.zeros(3)}
        m2 = {"w": torch.ones(3)}
        m3 = {"w": torch.full((3,), 2.0)}
        merged = ModeConnect._merge([m1, m2, m3], torch.tensor([0.5, 0.5, 0.0]))
        assert torch.allclose(merged["w"], torch.full((3,), 0.5))

    def test_int_buffers_passthrough(self):
        m1 = {"n": torch.tensor(7)}
        m2 = {"n": torch.tensor(9)}
        m3 = {"n": torch.tensor(11)}
        merged = ModeConnect._merge([m1, m2, m3], torch.tensor([0.2, 0.3, 0.5]))
        assert merged["n"].item() == 7  # int buffers come from vertex 0, unweighted


class TestModeConnectSmoke:
    def test_train_all_s_tier(self):
        config = ModeConnectConfig(
            hidden_dim=8, fast_demo=True, data_root=DATA_ROOT, epochs=1, n_grid=4,
            n_ensemble=2,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_mode_connect")
            result = ModeConnect().train_all(config, logger)
            assert 0.0 <= result["single_accuracy"] <= 1.0
            assert 0.0 <= result["ensemble_accuracy"] <= 1.0
            assert logger.artifact_path("loss_surface.png").exists()
            assert logger.artifact_path("simplex_vertices.pt").exists()
            keys = {m.get("key") for m in logger.read_metrics()}
            assert {"clf_a_loss", "clf_b_loss", "simplex_loss"} <= keys
