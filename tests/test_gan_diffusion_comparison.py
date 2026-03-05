"""Smoke tests for GAN vs Diffusion comparison composition."""
import os
import tempfile

import torch
import pytest

from mini_networks.compositions.gan_diffusion_comparison import (
    GANDiffusionComparison,
    GANDiffusionConfig,
    _pixel_variance,
)
from mini_networks.core.logging.logger import Logger

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


def _cfg(**kwargs):
    defaults = dict(
        image_size=28, latent_dim=16,
        base_channels=8, timesteps=5,
        fast_demo=True, data_root=DATA_ROOT, epochs=1,
    )
    defaults.update(kwargs)
    return GANDiffusionConfig(**defaults)


class TestGANDiffusionConfig:
    def test_defaults(self):
        cfg = GANDiffusionConfig()
        assert cfg.image_size == 28
        assert cfg.latent_dim == 100

    def test_fast_demo(self):
        cfg = GANDiffusionConfig(fast_demo=True)
        assert cfg.effective_epochs == 1


class TestPixelVariance:
    def test_output_is_float(self):
        imgs = torch.randn(4, 1, 28, 28)
        v = _pixel_variance(imgs)
        assert isinstance(v, float)

    def test_constant_image_has_zero_variance(self):
        imgs = torch.ones(4, 1, 28, 28)
        assert _pixel_variance(imgs) == pytest.approx(0.0, abs=1e-5)

    def test_higher_noise_higher_variance(self):
        low = torch.randn(8, 1, 28, 28) * 0.1
        high = torch.randn(8, 1, 28, 28) * 2.0
        assert _pixel_variance(high) > _pixel_variance(low)


class TestGANDiffusionTraining:
    def test_train_gan_smoke(self):
        config = _cfg()
        comp = GANDiffusionComparison()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_gan")
            comp.train_gan(config, logger)
            assert comp.generator is not None
            assert comp.discriminator is not None

    def test_train_gan_checkpoints(self):
        config = _cfg()
        comp = GANDiffusionComparison()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_gan_ckpt")
            comp.train_gan(config, logger)
            assert logger.artifact_path("gan_generator.pt").exists()
            assert logger.artifact_path("gan_discriminator.pt").exists()

    def test_train_gan_metrics(self):
        config = _cfg()
        comp = GANDiffusionComparison()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_gan_metrics")
            comp.train_gan(config, logger)
            metrics = logger.read_metrics()
            keys = {m["key"] for m in metrics}
            assert "gan_d_loss" in keys
            assert "gan_g_loss" in keys

    def test_train_diffusion_smoke(self):
        config = _cfg()
        comp = GANDiffusionComparison()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_diff")
            comp.train_diffusion(config, logger)
            assert comp.unet is not None
            assert comp.scheduler is not None

    def test_train_diffusion_checkpoint(self):
        config = _cfg()
        comp = GANDiffusionComparison()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_diff_ckpt")
            comp.train_diffusion(config, logger)
            assert logger.artifact_path("diffusion.pt").exists()

    def test_train_all_smoke(self):
        config = _cfg()
        comp = GANDiffusionComparison()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_all")
            comp.train_all(config, logger)
            metrics = logger.read_metrics()
            keys = {m["key"] for m in metrics}
            assert "gan_d_loss" in keys
            assert "diff_loss" in keys


class TestGANDiffusionSampling:
    def _trained_comp(self, config, tmpdir):
        comp = GANDiffusionComparison()
        logger = Logger(output_dir=tmpdir, run_name="train")
        comp.train_all(config, logger)
        return comp

    def test_sample_gan_shape(self):
        config = _cfg()
        with tempfile.TemporaryDirectory() as tmpdir:
            comp = self._trained_comp(config, tmpdir)
            samples = comp.sample_gan(config, n_samples=4)
            assert samples.shape == (4, 1, 28, 28)

    def test_sample_gan_range(self):
        config = _cfg()
        with tempfile.TemporaryDirectory() as tmpdir:
            comp = self._trained_comp(config, tmpdir)
            samples = comp.sample_gan(config, n_samples=4)
            assert samples.min() >= 0.0 - 1e-5
            assert samples.max() <= 1.0 + 1e-5

    def test_sample_diffusion_shape(self):
        config = _cfg()
        with tempfile.TemporaryDirectory() as tmpdir:
            comp = self._trained_comp(config, tmpdir)
            samples = comp.sample_diffusion(config, n_samples=4)
            assert samples.shape == (4, 1, 28, 28)

    def test_sample_diffusion_range(self):
        config = _cfg()
        with tempfile.TemporaryDirectory() as tmpdir:
            comp = self._trained_comp(config, tmpdir)
            samples = comp.sample_diffusion(config, n_samples=4)
            assert samples.min() >= 0.0 - 1e-5
            assert samples.max() <= 1.0 + 1e-5

    def test_sample_gan_seeded_deterministic(self):
        config = _cfg()
        with tempfile.TemporaryDirectory() as tmpdir:
            comp = self._trained_comp(config, tmpdir)
            s1 = comp.sample_gan(config, n_samples=2, seed=0)
            s2 = comp.sample_gan(config, n_samples=2, seed=0)
            assert torch.allclose(s1, s2)

    def test_compare_returns_both(self):
        config = _cfg()
        with tempfile.TemporaryDirectory() as tmpdir:
            comp = self._trained_comp(config, tmpdir)
            result = comp.compare(config, n_samples=4, seed=42)
            assert "gan_samples" in result
            assert "diffusion_samples" in result
            assert result["gan_samples"].shape == (4, 1, 28, 28)
            assert result["diffusion_samples"].shape == (4, 1, 28, 28)

    def test_compare_diversity_metrics(self):
        config = _cfg()
        with tempfile.TemporaryDirectory() as tmpdir:
            comp = self._trained_comp(config, tmpdir)
            result = comp.compare(config, n_samples=4)
            assert "gan_diversity" in result
            assert "diffusion_diversity" in result
            assert isinstance(result["gan_diversity"], float)
            assert isinstance(result["diffusion_diversity"], float)

    def test_compare_mean_pixel_in_range(self):
        config = _cfg()
        with tempfile.TemporaryDirectory() as tmpdir:
            comp = self._trained_comp(config, tmpdir)
            result = comp.compare(config, n_samples=4)
            assert 0.0 <= result["gan_mean_pixel"] <= 1.0
            assert 0.0 <= result["diffusion_mean_pixel"] <= 1.0
