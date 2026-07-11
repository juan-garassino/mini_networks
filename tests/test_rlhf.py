"""Smoke tests for RewardModel, shakespearean_score, and RLHFTrainer."""
import os
import tempfile

import torch
import pytest

from mini_networks.models.rlhf.config import RLHFConfig
from mini_networks.models.rlhf.model import RewardModel, shakespearean_score
from mini_networks.models.rlhf.trainer import RLHFTrainer, make_rlhf_dataloader
from mini_networks.models.transformer.model import TransformerLM
from mini_networks.core.logging.logger import Logger

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


# ---------------------------------------------------------------------------
# Heuristic reward
# ---------------------------------------------------------------------------

class TestShakespeareanScore:
    def test_empty_string_returns_zero(self):
        assert shakespearean_score("") == 0.0

    def test_gibberish_scores_near_zero(self):
        # The reward is deliberately DENSE (see shakespearean_score): plain
        # English scores low-but-nonzero so PPO has a gradient; only true
        # gibberish bottoms out.
        assert shakespearean_score("xqzt blorp fneep grix") == 0.0

    def test_graded_bands_are_ordered(self):
        gibberish = shakespearean_score("xqzt blorp fneep grix")
        english = shakespearean_score("the dog ran fast over the hill")
        archaic = shakespearean_score("thou art a verily noble soul")
        assert gibberish < english < archaic

    def test_archaic_words_raise_score(self):
        score = shakespearean_score("thou art a verily noble soul")
        assert score > 0.0

    def test_all_archaic_returns_high_score(self):
        score = shakespearean_score("thou thee thy dost hath")
        assert score > 0.5

    def test_score_is_normalised(self):
        """Score should not exceed 1.0."""
        score = shakespearean_score("thou thou thou thou")
        assert score <= 1.0

    def test_case_insensitive(self):
        lower = shakespearean_score("thou art noble")
        upper = shakespearean_score("THOU ART NOBLE")
        assert lower == pytest.approx(upper, abs=1e-6)


# ---------------------------------------------------------------------------
# RewardModel
# ---------------------------------------------------------------------------

class TestRewardModel:
    def _base_lm(self):
        return TransformerLM(vocab_size=32, d_model=16, n_heads=2, n_layers=1, d_ff=32, seq_len=16)

    def test_forward_shape(self):
        lm = self._base_lm()
        rm = RewardModel(lm, hidden=8)
        tokens = torch.randint(0, 32, (2, 8))
        rewards = rm(tokens)
        assert rewards.shape == (2,)

    def test_base_lm_frozen(self):
        lm = self._base_lm()
        rm = RewardModel(lm, hidden=8)
        for p in rm.base_lm.parameters():
            assert not p.requires_grad

    def test_reward_head_trainable(self):
        lm = self._base_lm()
        rm = RewardModel(lm, hidden=8)
        for p in rm.reward_head.parameters():
            assert p.requires_grad

    def test_bradley_terry_loss_positive(self):
        lm = self._base_lm()
        rm = RewardModel(lm, hidden=8)
        chosen   = torch.randint(0, 32, (2, 8))
        rejected = torch.randint(0, 32, (2, 8))
        loss = rm.bradley_terry_loss(chosen, rejected)
        assert loss.item() > 0

    def test_bradley_terry_loss_backprop(self):
        lm = self._base_lm()
        rm = RewardModel(lm, hidden=8)
        chosen   = torch.randint(0, 32, (2, 8))
        rejected = torch.randint(0, 32, (2, 8))
        loss = rm.bradley_terry_loss(chosen, rejected)
        loss.backward()
        for p in rm.reward_head.parameters():
            assert p.grad is not None

    def test_no_nan(self):
        lm = self._base_lm()
        rm = RewardModel(lm, hidden=8)
        tokens = torch.randint(0, 32, (4, 12))
        rewards = rm(tokens)
        assert not torch.isnan(rewards).any()


# ---------------------------------------------------------------------------
# RLHFTrainer
# ---------------------------------------------------------------------------

class TestRLHFTrainer:
    def _config(self, **kwargs):
        defaults = dict(
            d_model=32, n_layers=1, n_heads=2, d_ff=64,
            seq_len=16, fast_demo=True, data_root=DATA_ROOT,
            epochs=1, n_rollouts=4, rollout_max_new=8,
            n_ppo_iters=2, ppo_epochs=1,
        )
        defaults.update(kwargs)
        return RLHFConfig(**defaults)

    def test_train_smoke(self):
        config = self._config()
        trainer = RLHFTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_rlhf")
            dl = make_rlhf_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            assert len(metrics) > 0

    def test_pretrain_and_ppo_metrics_logged(self):
        config = self._config()
        trainer = RLHFTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_metrics")
            dl = make_rlhf_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            keys = {m["key"] for m in logger.read_metrics()}
            assert "pretrain_loss" in keys
            assert "ppo_loss" in keys
            assert "avg_reward" in keys

    def test_checkpoint_saved(self):
        config = self._config()
        trainer = RLHFTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_ckpt")
            dl = make_rlhf_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            assert logger.artifact_path("model.pt").exists()

    def test_evaluate_returns_loss(self):
        config = self._config()
        trainer = RLHFTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_eval")
            dl = make_rlhf_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.evaluate(config, dl, logger)
            assert "eval_loss" in result

    def test_infer_returns_generated(self):
        config = self._config()
        trainer = RLHFTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_infer")
            dl = make_rlhf_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.infer(config, {"prompt": "KING", "max_new_tokens": 8})
            assert "generated" in result
            assert "reward" in result
            assert isinstance(result["generated"], str)
            assert 0.0 <= result["reward"] <= 1.0

    def test_ref_model_frozen(self):
        config = self._config()
        trainer = RLHFTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_ref")
            dl = make_rlhf_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            assert trainer.ref_model is not None
            for p in trainer.ref_model.parameters():
                assert not p.requires_grad
