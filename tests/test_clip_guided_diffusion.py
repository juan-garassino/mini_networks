"""Smoke tests for CLIP-guided diffusion composition + VAE + ConditionedUNet."""
import os
import tempfile

import torch
import pytest

from mini_networks.models.diffusion.model import ConditionedUNet, EmbedFC
from mini_networks.models.diffusion.vae import VAE, vae_loss
from mini_networks.models.diffusion.scheduler import NoiseScheduler
from mini_networks.compositions.clip_guided_diffusion import (
    CLIPGuidedDiffusion,
    CLIPGuidedDiffusionConfig,
)
from mini_networks.core.logging.logger import Logger

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


class TestEmbedFC:
    def test_shape(self):
        fc = EmbedFC(10, 64)
        x = torch.zeros(4, 10)
        out = fc(x)
        assert out.shape == (4, 64)

    def test_one_hot_input(self):
        fc = EmbedFC(10, 32)
        x = torch.eye(4, 10)   # 4 different one-hots
        out = fc(x)
        assert not torch.isnan(out).any()


class TestConditionedUNet:
    def test_forward_with_class(self):
        model = ConditionedUNet(in_channels=1, n_feat=16, n_classes=10)
        x = torch.randn(2, 1, 28, 28)
        t = torch.tensor([10, 50])
        c = torch.tensor([3, 7])
        out = model(x, t, c)
        assert out.shape == (2, 1, 28, 28)

    def test_forward_unconditional(self):
        model = ConditionedUNet(in_channels=1, n_feat=16, n_classes=10)
        x = torch.randn(2, 1, 28, 28)
        t = torch.tensor([10, 50])
        mask = torch.ones(2, dtype=torch.long)   # all unconditional
        out = model(x, t, torch.zeros(2, dtype=torch.long), mask)
        assert out.shape == (2, 1, 28, 28)

    def test_no_nan(self):
        model = ConditionedUNet(in_channels=1, n_feat=16, n_classes=10)
        x = torch.randn(3, 1, 28, 28)
        t = torch.randint(0, 100, (3,))
        c = torch.randint(0, 10, (3,))
        out = model(x, t, c)
        assert not torch.isnan(out).any()

    def test_cfg_doubles_batch(self):
        """CFG inference doubles the batch: cond + uncond."""
        model = ConditionedUNet(in_channels=1, n_feat=16, n_classes=10)
        B = 2
        x = torch.randn(B * 2, 1, 28, 28)    # doubled
        t = torch.randint(0, 100, (B * 2,))
        c = torch.tensor([3, 7, 3, 7])        # same labels
        mask = torch.tensor([0, 0, 1, 1])     # first half cond, second uncond
        out = model(x, t, c, mask)
        assert out.shape == (B * 2, 1, 28, 28)


class TestVAE:
    def test_encode_shape(self):
        vae = VAE(latent_channels=4)
        x = torch.randn(2, 1, 28, 28)
        mu, logvar = vae.encode(x)
        assert mu.shape == (2, 4, 7, 7)
        assert logvar.shape == (2, 4, 7, 7)

    def test_decode_shape(self):
        vae = VAE(latent_channels=4)
        z = torch.randn(2, 4, 7, 7)
        out = vae.decode(z)
        assert out.shape == (2, 1, 28, 28)

    def test_forward_roundtrip_shape(self):
        vae = VAE(latent_channels=4)
        x = torch.randn(2, 1, 28, 28)
        recon, mu, logvar = vae(x)
        assert recon.shape == x.shape

    def test_output_range(self):
        vae = VAE(latent_channels=4)
        vae.eval()
        z = torch.randn(2, 4, 7, 7)
        out = vae.decode(z)
        assert out.min() >= -1.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5

    def test_vae_loss(self):
        vae = VAE(latent_channels=4)
        x = torch.randn(2, 1, 28, 28)
        recon, mu, logvar = vae(x)
        loss = vae_loss(recon, x, mu, logvar)
        assert loss.item() > 0
        assert not torch.isnan(loss)


class TestCLIPGuidedDiffusionConfig:
    def test_defaults(self):
        cfg = CLIPGuidedDiffusionConfig()
        assert cfg.n_classes == 10
        assert cfg.guide_weight == 2.0
        assert cfg.flip_every == 50

    def test_fast_demo(self):
        cfg = CLIPGuidedDiffusionConfig(fast_demo=True)
        assert cfg.effective_epochs == 1


class TestCLIPGuidedDiffusionPipeline:
    def _make_config(self):
        return CLIPGuidedDiffusionConfig(
            embed_dim=16,
            vocab_size=64,
            text_seq_len=8,
            text_d_model=16,
            text_n_heads=2,
            text_n_layers=1,
            n_feat=16,
            timesteps=5,          # tiny for speed
            fast_demo=True,
            data_root=DATA_ROOT,
            epochs=1,
            guide_weight=1.0,
            flip_every=2,
        )

    def test_train_clip(self):
        config = self._make_config()
        comp = CLIPGuidedDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test")
            comp.train_clip(config, logger)
            assert comp.clip is not None
            assert len(comp._class_text_embeds) == 10

    def test_train_diffusion(self):
        config = self._make_config()
        comp = CLIPGuidedDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test")
            comp.train_diffusion(config, logger)
            assert comp.unet is not None
            assert comp.scheduler is not None

    def test_sample(self):
        config = self._make_config()
        comp = CLIPGuidedDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test")
            comp.train_diffusion(config, logger)
            samples = comp.sample(class_id=3, n_samples=2, config=config)
            assert samples.shape == (2, 1, 28, 28)
            assert samples.min() >= 0.0 and samples.max() <= 1.0

    def test_text_to_class(self):
        config = self._make_config()
        comp = CLIPGuidedDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test")
            comp.train_clip(config, logger)
            cls_id = comp.text_to_class("three", config)
            assert 0 <= cls_id <= 9

    def test_text_to_image(self):
        config = self._make_config()
        comp = CLIPGuidedDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test")
            comp.train_all(config, logger)
            images, cls_id = comp.text_to_image("five", config, n_samples=2)
            assert images.shape == (2, 1, 28, 28)
            assert 0 <= cls_id <= 9

    def test_dual_oscillation_shape(self):
        config = self._make_config()
        comp = CLIPGuidedDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test")
            comp.train_diffusion(config, logger)
            final = comp.dual_oscillation(class_a=3, class_b=8, config=config)
            assert final.shape == (1, 1, 28, 28)
            assert final.min() >= 0.0 and final.max() <= 1.0

    def test_dual_oscillation_frames(self):
        config = self._make_config()
        comp = CLIPGuidedDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test")
            comp.train_diffusion(config, logger)
            final, frames = comp.dual_oscillation(
                class_a=0, class_b=1, config=config, return_frames=True
            )
            assert len(frames) > 0
            assert frames[0].shape[-2:] == (28, 28)

    def test_vae_pipeline(self):
        config = self._make_config()
        config = config.model_copy(update={"use_vae": True})
        comp = CLIPGuidedDiffusion()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test")
            comp.train_vae(config, logger)
            comp.train_diffusion(config, logger)
            samples = comp.sample(class_id=5, n_samples=2, config=config)
            assert samples.shape == (2, 1, 28, 28)
