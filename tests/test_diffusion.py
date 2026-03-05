"""Diffusion: scheduler, UNet forward, trainer 1-step smoke test + EMA/curriculum."""
import os
import tempfile

import pytest
import torch

from mini_networks.models.diffusion.config import DiffusionConfig
from mini_networks.models.diffusion.model import UNet
from mini_networks.models.diffusion.scheduler import NoiseScheduler
from mini_networks.models.diffusion.trainer import (
    DDPMTrainer,
    EMA,
    _image_complexity,
    _sort_batch_by_complexity,
    make_diffusion_dataloader,
)
from mini_networks.core.logging.logger import Logger

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


class TestNoiseScheduler:
    def test_add_noise_shape(self):
        sched = NoiseScheduler(timesteps=100)
        x0 = torch.randn(2, 1, 28, 28)
        noise = torch.randn_like(x0)
        t = torch.tensor([10, 50])
        xt = sched.add_noise(x0, noise, t)
        assert xt.shape == x0.shape

    def test_step_output_shape(self):
        sched = NoiseScheduler(timesteps=10)
        model_out = torch.randn(2, 1, 28, 28)
        x_t = torch.randn(2, 1, 28, 28)
        x_prev = sched.step(model_out, 5, x_t)
        assert x_prev.shape == x_t.shape

    def test_cosine_schedule(self):
        sched = NoiseScheduler(timesteps=100, schedule="cosine")
        assert sched.betas.shape == (100,)
        assert (sched.betas > 0).all()


class TestUNet:
    def test_forward_shape(self):
        model = UNet(in_channels=1, base_channels=16)
        x = torch.randn(2, 1, 28, 28)
        t = torch.tensor([10, 50])
        out = model(x, t)
        assert out.shape == (2, 1, 28, 28)

    def test_forward_no_nan(self):
        model = UNet(in_channels=1, base_channels=16)
        x = torch.randn(2, 1, 28, 28)
        t = torch.tensor([0, 999])
        out = model(x, t)
        assert not torch.isnan(out).any()


class TestDDPMTrainer:
    def test_train_smoke(self):
        config = DiffusionConfig(
            timesteps=10,
            base_channels=8,
            fast_demo=True,
            data_root=DATA_ROOT,
            epochs=1,
        )
        trainer = DDPMTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test")
            dl = make_diffusion_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            assert len(metrics) > 0

    def test_infer_returns_samples(self):
        config = DiffusionConfig(
            timesteps=5, base_channels=8, fast_demo=True, data_root=DATA_ROOT, epochs=1,
        )
        trainer = DDPMTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test")
            dl = make_diffusion_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.infer(config, {"n_samples": 2})
            assert "samples" in result
            assert result["samples"].shape == (2, 1, 28, 28)


class TestEMA:
    def test_shadow_copy_created(self):
        model = UNet(in_channels=1, base_channels=8)
        ema = EMA(model, decay=0.999)
        # Shadow should have same parameter count
        assert len(list(ema.shadow.parameters())) == len(list(model.parameters()))

    def test_shadow_params_not_require_grad(self):
        model = UNet(in_channels=1, base_channels=8)
        ema = EMA(model, decay=0.999)
        assert all(not p.requires_grad for p in ema.shadow.parameters())

    def test_update_changes_shadow(self):
        model = UNet(in_channels=1, base_channels=8)
        ema = EMA(model, decay=0.9)
        # Record initial shadow value
        with torch.no_grad():
            initial = list(ema.shadow.parameters())[0].clone()
        # Change model weight
        with torch.no_grad():
            list(model.parameters())[0].fill_(99.0)
        ema.update(model)
        updated = list(ema.shadow.parameters())[0]
        # Shadow should move toward 99.0
        assert not torch.allclose(updated, initial)

    def test_update_decay_zero_copies_exactly(self):
        """decay=0.0 means shadow = model params exactly after one update."""
        model = UNet(in_channels=1, base_channels=8)
        ema = EMA(model, decay=0.0)
        param = list(model.parameters())[0]
        with torch.no_grad():
            param.fill_(42.0)
        ema.update(model)
        shadow_param = list(ema.shadow.parameters())[0]
        assert shadow_param.flatten()[0].item() == pytest.approx(42.0, abs=1e-5)

    def test_state_dict_returns_shadow_weights(self):
        model = UNet(in_channels=1, base_channels=8)
        ema = EMA(model, decay=0.999)
        sd = ema.state_dict()
        assert isinstance(sd, dict)
        assert len(sd) > 0


class TestCurriculum:
    def test_complexity_shape(self):
        images = torch.randn(8, 1, 28, 28)
        c = _image_complexity(images)
        assert c.shape == (8,)

    def test_sort_descending(self):
        images = torch.randn(8, 1, 28, 28)
        labels = torch.arange(8)
        sorted_imgs, sorted_lbls = _sort_batch_by_complexity(images, labels, descending=True)
        c_sorted = _image_complexity(sorted_imgs)
        # First should have highest complexity
        assert c_sorted[0] >= c_sorted[-1]

    def test_sort_ascending(self):
        images = torch.randn(8, 1, 28, 28)
        labels = torch.arange(8)
        sorted_imgs, sorted_lbls = _sort_batch_by_complexity(images, labels, descending=False)
        c_sorted = _image_complexity(sorted_imgs)
        assert c_sorted[0] <= c_sorted[-1]


class TestDDPMTrainerAdvanced:
    def _config(self, **kwargs):
        defaults = dict(
            timesteps=5, base_channels=8, fast_demo=True,
            data_root=DATA_ROOT, epochs=1,
        )
        defaults.update(kwargs)
        return DiffusionConfig(**defaults)

    def test_ema_checkpoint_saved(self):
        config = self._config(ema_decay=0.9)
        trainer = DDPMTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_ema")
            dl = make_diffusion_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            assert logger.artifact_path("model_ema.pt").exists()

    def test_ema_disabled_no_extra_checkpoint(self):
        config = self._config(ema_decay=0.0)
        trainer = DDPMTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_no_ema")
            dl = make_diffusion_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            assert not logger.artifact_path("model_ema.pt").exists()

    def test_curriculum_train_smoke(self):
        config = self._config(curriculum=True)
        trainer = DDPMTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_curriculum")
            dl = make_diffusion_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            assert len(metrics) > 0

    def test_warmup_train_smoke(self):
        config = self._config(warmup_steps=5)
        trainer = DDPMTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_warmup")
            dl = make_diffusion_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            assert len(metrics) > 0

    def test_ema_infer_uses_shadow(self):
        config = self._config(ema_decay=0.9)
        trainer = DDPMTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_ema_infer")
            dl = make_diffusion_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.infer(config, {"n_samples": 2})
            assert result["samples"].shape == (2, 1, 28, 28)
