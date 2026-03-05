"""Smoke tests for Transformer → CLIP → Diffusion composition."""
import os
import tempfile

import torch
import pytest

from mini_networks.compositions.transformer_clip_diffusion import (
    TransformerCLIPDiffusion,
    TransformerCLIPDiffusionConfig,
)
from mini_networks.core.logging.logger import Logger

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


def _cfg(**kwargs):
    defaults = dict(
        lm_d_model=32, lm_n_heads=2, lm_n_layers=1, lm_d_ff=64, lm_seq_len=16,
        embed_dim=16, vocab_size=64, text_seq_len=8,
        text_d_model=16, text_n_heads=2, text_n_layers=1,
        n_feat=16, timesteps=5,
        fast_demo=True, data_root=DATA_ROOT, epochs=1,
        k_prompts=3, prompt_max_new=8,
    )
    defaults.update(kwargs)
    return TransformerCLIPDiffusionConfig(**defaults)


class TestTransformerCLIPDiffusionConfig:
    def test_defaults(self):
        cfg = TransformerCLIPDiffusionConfig()
        assert cfg.n_classes == 10
        assert cfg.k_prompts == 8

    def test_fast_demo(self):
        cfg = TransformerCLIPDiffusionConfig(fast_demo=True)
        assert cfg.effective_epochs == 1


class TestTransformerCLIPDiffusionStages:
    def test_train_lm(self):
        config = _cfg()
        comp = TransformerCLIPDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_lm")
            comp.train_lm(config, logger)
            assert comp.lm is not None
            assert comp.tokenizer is not None
            assert logger.artifact_path("lm.pt").exists()

    def test_train_clip(self):
        config = _cfg()
        comp = TransformerCLIPDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_clip")
            comp.train_clip(config, logger)
            assert comp.clip is not None
            assert len(comp._class_embeds) == 10
            assert logger.artifact_path("clip.pt").exists()

    def test_train_diffusion(self):
        config = _cfg()
        comp = TransformerCLIPDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_diff")
            comp.train_diffusion(config, logger)
            assert comp.unet is not None
            assert comp.scheduler is not None
            assert logger.artifact_path("unet.pt").exists()

    def test_train_all_smoke(self):
        config = _cfg()
        comp = TransformerCLIPDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_all")
            comp.train_all(config, logger)
            metrics = logger.read_metrics()
            keys = {m["key"] for m in metrics}
            assert "lm_loss" in keys
            assert "clip_loss" in keys
            assert "diff_loss" in keys

    def test_train_all_checkpoints(self):
        config = _cfg()
        comp = TransformerCLIPDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_ckpts")
            comp.train_all(config, logger)
            assert logger.artifact_path("lm.pt").exists()
            assert logger.artifact_path("clip.pt").exists()
            assert logger.artifact_path("unet.pt").exists()


class TestTransformerCLIPDiffusionInference:
    def _trained_comp(self, config):
        comp = TransformerCLIPDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="train")
            comp.train_all(config, logger)
        return comp

    def test_generate_prompts_count(self):
        config = _cfg(k_prompts=4)
        comp = TransformerCLIPDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="p")
            comp.train_lm(config, logger)
        prompts = comp.generate_prompts("KING", config)
        assert len(prompts) == 4

    def test_generate_prompts_are_strings(self):
        config = _cfg(k_prompts=3)
        comp = TransformerCLIPDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="p")
            comp.train_lm(config, logger)
        prompts = comp.generate_prompts("hello", config)
        assert all(isinstance(p, str) for p in prompts)

    def test_rank_prompts_returns_valid_class(self):
        config = _cfg(k_prompts=3)
        comp = TransformerCLIPDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="rank")
            comp.train_lm(config, logger)
            comp.train_clip(config, logger)
        prompts = comp.generate_prompts("five", config)
        class_id, scores = comp.rank_prompts_by_class(prompts, config)
        assert 0 <= class_id <= 9
        assert len(scores) == 10

    def test_sample_class_shape(self):
        config = _cfg()
        comp = TransformerCLIPDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="samp")
            comp.train_diffusion(config, logger)
        samples = comp.sample_class(class_id=3, config=config, n_samples=2)
        assert samples.shape == (2, 1, 28, 28)
        assert samples.min() >= 0.0 and samples.max() <= 1.0

    def test_generate_image_full_pipeline(self):
        config = _cfg(k_prompts=3)
        comp = TransformerCLIPDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="full")
            comp.train_all(config, logger)
        images, class_id, prompts = comp.generate_image("three", config, n_samples=2)
        assert images.shape == (2, 1, 28, 28)
        assert 0 <= class_id <= 9
        assert len(prompts) == config.k_prompts

    def test_generate_image_pixel_range(self):
        config = _cfg(k_prompts=3)
        comp = TransformerCLIPDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="range")
            comp.train_all(config, logger)
        images, _, _ = comp.generate_image("seven", config, n_samples=2)
        assert images.min() >= 0.0 - 1e-5
        assert images.max() <= 1.0 + 1e-5

    def test_class_embeds_are_normalized(self):
        config = _cfg()
        comp = TransformerCLIPDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="emb")
            comp.train_clip(config, logger)
        for cls_id, emb in comp._class_embeds.items():
            norm = emb.norm().item()
            assert abs(norm - 1.0) < 1e-4, f"class {cls_id} embed norm {norm:.4f} != 1.0"
