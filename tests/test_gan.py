"""Smoke tests for GAN: Generator, Discriminator, losses, trainer."""
import os
import tempfile

import torch
import torch.nn as nn

from mini_networks.models.gan.config import GANConfig
from mini_networks.models.gan.model import (
    Discriminator,
    Generator,
    gan_d_loss,
    gan_g_loss,
)
from mini_networks.models.gan.trainer import GANTrainer, make_gan_dataloader
from mini_networks.core.logging.logger import Logger

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class TestGenerator:
    def test_output_shape(self):
        G = Generator(latent_dim=16, image_size=28, in_channels=1)
        z = torch.randn(4, 16)
        out = G(z)
        assert out.shape == (4, 1, 28, 28)

    def test_output_range(self):
        """Tanh output must be in [-1, 1]."""
        G = Generator(latent_dim=16, image_size=28, in_channels=1)
        G.eval()
        z = torch.randn(8, 16)
        out = G(z)
        assert out.min() >= -1.0 - 1e-5
        assert out.max() <=  1.0 + 1e-5

    def test_no_nan(self):
        G = Generator(latent_dim=100)
        z = torch.randn(8, 100)
        assert not torch.isnan(G(z)).any()

    def test_different_noise_different_output(self):
        G = Generator(latent_dim=16)
        G.eval()
        z1 = torch.randn(2, 16)
        z2 = torch.randn(2, 16)
        assert not torch.allclose(G(z1), G(z2))

    def test_backprop(self):
        G = Generator(latent_dim=16)
        z = torch.randn(4, 16)
        G(z).mean().backward()
        for p in G.parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class TestDiscriminator:
    def test_output_shape(self):
        D = Discriminator(image_size=28, in_channels=1)
        x = torch.randn(4, 1, 28, 28)
        out = D(x)
        assert out.shape == (4, 1)

    def test_output_range(self):
        """Sigmoid output must be in [0, 1]."""
        D = Discriminator(image_size=28, in_channels=1)
        D.eval()
        x = torch.randn(8, 1, 28, 28)
        out = D(x)
        assert out.min() >= 0.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5

    def test_no_nan(self):
        D = Discriminator()
        x = torch.randn(4, 1, 28, 28)
        assert not torch.isnan(D(x)).any()

    def test_backprop(self):
        D = Discriminator()
        x = torch.randn(4, 1, 28, 28)
        D(x).mean().backward()
        for p in D.parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class TestGANLosses:
    def setup_method(self):
        self.G = Generator(latent_dim=16, image_size=28, in_channels=1)
        self.D = Discriminator(image_size=28, in_channels=1)
        self.criterion = nn.BCELoss()

    def test_d_loss_positive(self):
        real = torch.randn(4, 1, 28, 28)
        z = torch.randn(4, 16)
        fake = self.G(z)
        loss = gan_d_loss(self.D, real, fake, self.criterion)
        assert loss.item() > 0

    def test_g_loss_positive(self):
        z = torch.randn(4, 16)
        fake = self.G(z)
        loss = gan_g_loss(self.D, fake, self.criterion)
        assert loss.item() > 0

    def test_d_loss_detaches_fake(self):
        """gan_d_loss must not propagate gradients to G via the fake branch."""
        real = torch.randn(4, 1, 28, 28)
        z = torch.randn(4, 16, requires_grad=True)
        fake = self.G(z)
        loss = gan_d_loss(self.D, real, fake, self.criterion)
        loss.backward()
        # Generator parameters should have no gradient from D loss
        for p in self.G.parameters():
            assert p.grad is None

    def test_g_loss_propagates_to_g(self):
        """gan_g_loss must propagate gradients to G."""
        z = torch.randn(4, 16)
        fake = self.G(z)
        loss = gan_g_loss(self.D, fake, self.criterion)
        loss.backward()
        for p in self.G.parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TestGANTrainer:
    def _config(self, **kwargs):
        defaults = dict(
            latent_dim=8,
            fast_demo=True,
            data_root=DATA_ROOT,
            epochs=1,
            lr=0.0002,
        )
        defaults.update(kwargs)
        return GANConfig(**defaults)

    def test_train_smoke(self):
        config = self._config()
        trainer = GANTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_gan")
            dl = make_gan_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            assert len(metrics) > 0
            keys = {m["key"] for m in metrics}
            assert "d_loss" in keys
            assert "g_loss" in keys

    def test_checkpoints_saved(self):
        config = self._config()
        trainer = GANTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_gan")
            dl = make_gan_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            assert logger.artifact_path("generator.pt").exists()
            assert logger.artifact_path("discriminator.pt").exists()

    def test_evaluate(self):
        config = self._config()
        trainer = GANTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_gan")
            dl = make_gan_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.evaluate(config, dl, logger)
            assert "mean_real_score" in result
            assert 0.0 <= result["mean_real_score"] <= 1.0

    def test_infer_shape(self):
        config = self._config()
        trainer = GANTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_gan")
            dl = make_gan_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.infer(config, {"n_samples": 5, "seed": 0})
            samples = result["samples"]
            assert samples.shape == (5, 1, 28, 28)

    def test_infer_range(self):
        """Samples should be normalised to [0, 1] after Tanh rescaling."""
        config = self._config()
        trainer = GANTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_gan")
            dl = make_gan_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.infer(config, {"n_samples": 4})
            s = result["samples"]
            assert s.min() >= 0.0 - 1e-5
            assert s.max() <= 1.0 + 1e-5

    def test_infer_seeded_deterministic(self):
        """Same seed should produce identical samples."""
        config = self._config()
        trainer = GANTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_gan")
            dl = make_gan_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            r1 = trainer.infer(config, {"n_samples": 3, "seed": 7})
            r2 = trainer.infer(config, {"n_samples": 3, "seed": 7})
            assert torch.allclose(r1["samples"], r2["samples"])
