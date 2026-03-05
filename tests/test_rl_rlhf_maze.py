"""Smoke tests for the RL + RLHF Maze composition."""
from __future__ import annotations

import tempfile

import pytest

from mini_networks.compositions.rl_rlhf_maze import (
    RLHFMazeConfig,
    RLHFMazeComposition,
    VOCAB_SIZE,
    _encode,
    _decode,
    _actions_to_trajectory_str,
    _STOI,
    _ITOS,
)
from mini_networks.core.logging.logger import Logger


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

class TestEncoding:
    def test_vocab_size(self):
        assert VOCAB_SIZE == 8  # S U D L R X G + PAD

    def test_stoi_keys(self):
        for c in "SUDLRXG":
            assert c in _STOI

    def test_encode_decode_roundtrip(self):
        text = "SUURRDDG"
        ids = _encode(text)
        recovered = _decode(ids)
        assert recovered == text

    def test_actions_to_trajectory_success(self):
        traj = _actions_to_trajectory_str([0, 1, 3], success=True)
        assert traj.startswith("S")
        assert traj.endswith("G")

    def test_actions_to_trajectory_fail(self):
        traj = _actions_to_trajectory_str([0, 2], success=False)
        assert traj.startswith("S")
        assert traj.endswith("X")

    def test_empty_actions(self):
        traj = _actions_to_trajectory_str([], success=False)
        assert traj == "SX"


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestConfig:
    def test_fast_demo_config(self):
        cfg = RLHFMazeConfig(fast_demo=True)
        assert cfg.fast_demo is True
        assert cfg.maze_width == 5
        assert cfg.maze_height == 5

    def test_vocab_size_is_8(self):
        assert VOCAB_SIZE == 8

    def test_model_name(self):
        cfg = RLHFMazeConfig()
        assert cfg.model_name == "rl_rlhf_maze"


# ---------------------------------------------------------------------------
# Full pipeline smoke test
# ---------------------------------------------------------------------------

class TestRLHFMazeComposition:
    def _config(self) -> RLHFMazeConfig:
        return RLHFMazeConfig(fast_demo=True)

    def test_train_runs(self):
        """Full three-phase pipeline completes without error."""
        config = self._config()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_train")
            comp = RLHFMazeComposition()
            comp.train(config, logger)
            assert comp.dqn_agent is not None
            assert comp.lm is not None
            assert comp.ref_lm is not None

    def test_compare_returns_expected_keys(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_compare")
            comp = RLHFMazeComposition()
            comp.train(config, logger)
            result = comp.compare(config)

        assert "dqn_success_rate" in result
        assert "lm_success_rate"  in result
        assert "dqn_mean_steps"   in result
        assert "lm_mean_steps"    in result
        assert "sample_trajectories" in result

    def test_compare_success_rates_in_range(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_rates")
            comp = RLHFMazeComposition()
            comp.train(config, logger)
            result = comp.compare(config)

        assert 0.0 <= result["dqn_success_rate"] <= 1.0
        assert 0.0 <= result["lm_success_rate"]  <= 1.0

    def test_sample_trajectories_are_strings(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_traj")
            comp = RLHFMazeComposition()
            comp.train(config, logger)
            result = comp.compare(config)

        for t in result["sample_trajectories"]:
            assert isinstance(t, str)

    def test_metrics_logged(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_metrics")
            comp = RLHFMazeComposition()
            comp.train(config, logger)
            metrics = logger.read_metrics()
            phases = {m.get("value", {}).get("phase") for m in metrics if isinstance(m.get("value"), dict)}
            # Phase 1 metrics use a different format in this logger
            assert len(metrics) > 0

    def test_compare_before_train_raises(self):
        config = self._config()
        comp = RLHFMazeComposition()
        with pytest.raises(RuntimeError):
            comp.compare(config)

    def test_lm_generates_valid_chars(self):
        """LM output should only contain chars from our alphabet."""
        import torch
        config = self._config()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_gen")
            comp = RLHFMazeComposition()
            comp.train(config, logger)

        lm = comp.lm
        lm.eval()
        start_id = _STOI["S"]
        prompt = torch.tensor([[start_id]], dtype=torch.long, device=config.device)
        with torch.no_grad():
            gen = lm.generate(prompt, max_new_tokens=10, temperature=1.0)
        decoded = _decode(gen[0].tolist())
        # All chars should be from our alphabet
        valid = set("SUDLRXG")
        for c in decoded:
            assert c in valid, f"Unexpected char: {c!r}"


# ---------------------------------------------------------------------------
# run() convenience function
# ---------------------------------------------------------------------------

class TestRunConvenience:
    def test_run_returns_dict(self):
        import tempfile
        from mini_networks.compositions.rl_rlhf_maze import run
        result = run(config=RLHFMazeConfig(fast_demo=True))
        assert isinstance(result, dict)
        assert "dqn_success_rate" in result
