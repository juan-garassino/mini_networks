"""Tests for mini NeRF: camera math, compositing, split hygiene, trainer."""
import os
import tempfile

import torch

from mini_networks.core.logging.logger import Logger
from mini_networks.models.nerf.config import NerfConfig
from mini_networks.models.nerf.model import MiniNeRF, render_rays
from mini_networks.models.nerf.scene import (
    NerfViewDataset,
    composite,
    make_nerf_dataloader,
    make_rays,
    render_gt,
    build_voxels,
)

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


def _config(**kwargs):
    defaults = dict(fast_demo=True, data_root=DATA_ROOT, epochs=1)
    defaults.update(kwargs)
    return NerfConfig(**defaults)


class TestCamera:
    def test_rays_look_at_center(self):
        """The central ray must point (anti-)parallel to the eye vector."""
        rays_o, rays_d = make_rays(azimuth_deg=0.0, elevation_deg=35.0, size=33, radius=2.5)
        center = rays_d[33 * 16 + 16]
        eye = rays_o[0]
        cos = torch.dot(center, -eye / eye.norm())
        assert cos > 0.999

    def test_rays_normalized(self):
        _, rays_d = make_rays(120.0, 35.0, 16, 2.5)
        assert torch.allclose(rays_d.norm(dim=-1), torch.ones(256), atol=1e-5)


class TestCompositing:
    def test_weights_bounded(self):
        """Accumulated color can never exceed the max sample color."""
        sigma = torch.rand(10, 32) * 50
        color = torch.ones(10, 32, 3)
        t = torch.linspace(1, 4, 32).expand(10, -1)
        rgb = composite(sigma, color, t)
        assert (rgb <= 1.0 + 1e-5).all() and (rgb >= 0).all()

    def test_empty_scene_is_black(self):
        sigma = torch.zeros(5, 16)
        color = torch.ones(5, 16, 3)
        t = torch.linspace(1, 4, 16).expand(5, -1)
        assert composite(sigma, color, t).abs().max() < 1e-6


class TestScene:
    def test_gt_deterministic(self):
        config = _config()
        occ, rgb = build_voxels(DATA_ROOT, config.seed, config.depth_layers)
        a = render_gt(occ, rgb, 30.0, config)
        b = render_gt(occ, rgb, 30.0, config)
        assert torch.equal(a, b)

    def test_gt_has_content(self):
        config = _config()
        occ, rgb = build_voxels(DATA_ROOT, config.seed, config.depth_layers)
        view = render_gt(occ, rgb, 0.0, config)
        assert (view.sum(-1) > 0.1).sum() > 50  # the digit is visible

    def test_split_disjoint_no_wraparound(self):
        config = _config()
        train = NerfViewDataset(config, "train")
        test = NerfViewDataset(config, "test")
        assert set(train.azimuths).isdisjoint(test.azimuths)
        assert 360.0 not in train.azimuths + test.azimuths  # no 0/360 duplicate
        assert len(test) > 0


class TestMiniNeRF:
    def test_field_shapes(self):
        model = MiniNeRF(pe_freqs=4, hidden_dim=32, n_layers=3)
        sigma, rgb = model(torch.randn(7, 5, 3))
        assert sigma.shape == (7, 5)
        assert rgb.shape == (7, 5, 3)
        assert (sigma >= 0).all() and (rgb >= 0).all() and (rgb <= 1).all()

    def test_eval_render_deterministic(self):
        """Bin midpoints at eval: identical calls, identical PSNR inputs."""
        torch.manual_seed(0)
        model = MiniNeRF(pe_freqs=4, hidden_dim=32, n_layers=3)
        rays_o, rays_d = make_rays(0.0, 35.0, 8, 2.5)
        a = render_rays(model, rays_o, rays_d, 16, 1.0, 4.0, stratified=False)
        b = render_rays(model, rays_o, rays_d, 16, 1.0, 4.0, stratified=False)
        assert torch.equal(a, b)


class TestNerfTrainer:
    def test_train_eval_infer_checkpoint(self):
        from mini_networks.models.nerf.trainer import NerfTrainer

        config = _config()
        trainer = NerfTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_nerf")
            trainer.train(config, make_nerf_dataloader(config, "train"), logger)
            assert len(logger.read_metrics()) > 0
            result = trainer.evaluate(config, make_nerf_dataloader(config, "test"), logger)
            assert "psnr" in result and "psnr_min" in result
            assert result["psnr_min"] <= result["psnr"]
            out = trainer.infer(config, {"azimuth": 45})
            assert out["image"].shape == (3, config.image_size, config.image_size)
            fresh = NerfTrainer()
            fresh.load_checkpoint(config, logger.artifact_path("model.pt").parent)
            assert fresh.model is not None
