"""Tests for mini SAM: composite data, prompt plumbing, min-loss semantics."""
import os
import tempfile

import torch

from mini_networks.core.data.registry import TwoDigitSamDataset
from mini_networks.core.logging.logger import Logger
from mini_networks.models.sam.config import SAMConfig
from mini_networks.models.sam.model import MiniSAM, dice_loss
from mini_networks.models.sam.trainer import (
    SAMTrainer,
    interior_click,
    make_sam_dataloader,
)

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


class TestTwoDigitSamDataset:
    def test_deterministic(self):
        a = TwoDigitSamDataset(DATA_ROOT, split="train", fast_demo=True)
        b = TwoDigitSamDataset(DATA_ROOT, split="train", fast_demo=True)
        ia, ma, _ = a[5]
        ib, mb, _ = b[5]
        assert torch.equal(ia, ib) and torch.equal(ma, mb)

    def test_two_masks_differ_and_compose(self):
        ds = TwoDigitSamDataset(DATA_ROOT, split="train", fast_demo=True)
        img, mask_a, mask_b = ds[0]
        assert img.shape == (1, 56, 56)
        assert not torch.equal(mask_a, mask_b)  # the ambiguity is real
        union = ((mask_a + mask_b) > 0).float()
        assert torch.equal((img.squeeze(0) > 0).float(), union)

    def test_split_seeds_differ(self):
        tr = TwoDigitSamDataset(DATA_ROOT, split="train", fast_demo=True)
        te = TwoDigitSamDataset(DATA_ROOT, split="test", fast_demo=True)
        assert not torch.equal(tr[0][0], te[0][0])


class TestInteriorClick:
    def test_click_lands_on_foreground(self):
        """A ring mask (COM in the hole) must snap to a foreground pixel."""
        mask = torch.zeros(56, 56)
        mask[10:20, 10:20] = 1.0
        mask[13:17, 13:17] = 0.0  # hole at the center of mass
        yx = (interior_click(mask) * 56).long()
        assert mask[yx[0], yx[1]] == 1.0


class TestMiniSAM:
    def test_forward_shapes(self):
        model = MiniSAM(embed_dim=32, n_heads=2)
        img = torch.randn(2, 1, 56, 56)
        coords = torch.rand(2, 2, 2)
        types = torch.tensor([[0, 1], [2, 3]])
        masks, iou = model(img, coords, types)
        assert masks.shape == (2, 3, 56, 56)
        assert iou.shape == (2, 3)

    def test_prompt_changes_output(self):
        """Different clicks on the same image must change the decoder output —
        the prompt path is wired in, even at init."""
        torch.manual_seed(0)
        model = MiniSAM(embed_dim=32, n_heads=2)
        model.eval()
        img = torch.randn(1, 1, 56, 56)
        t = torch.zeros(1, 1, dtype=torch.long)
        with torch.no_grad():
            m1, _ = model(img, torch.tensor([[[0.2, 0.2]]]), t)
            m2, _ = model(img, torch.tensor([[[0.8, 0.8]]]), t)
        assert not torch.allclose(m1, m2, atol=1e-5)

    def test_dice_per_sample(self):
        probs = torch.stack([torch.ones(8, 8), torch.zeros(8, 8)])
        target = torch.ones(2, 8, 8)
        d = dice_loss(probs, target)
        assert d.shape == (2,)
        assert d[0] < 0.01 and d[1] > 0.9


class TestSAMTrainer:
    def _config(self, **kwargs):
        defaults = dict(embed_dim=32, n_heads=2, fast_demo=True,
                        data_root=DATA_ROOT, epochs=1)
        defaults.update(kwargs)
        return SAMConfig(**defaults)

    def test_train_smoke(self):
        config = self._config()
        trainer = SAMTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_sam")
            trainer.train(config, make_sam_dataloader(config), logger)
            keys = {m.get("key") for m in logger.read_metrics()}
            assert "loss" in keys
            assert "head0_wins" in keys  # head-collapse visibility
            assert logger.artifact_path("model.pt").exists()

    def test_evaluate_reports_promptability(self):
        config = self._config()
        trainer = SAMTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_sam")
            trainer.train(config, make_sam_dataloader(config), logger)
            result = trainer.evaluate(config, make_sam_dataloader(config, split="test"), logger)
            assert 0.0 <= result["eval_iou"] <= 1.0
            assert 0.0 <= result["wrong_prompt_iou"] <= 1.0

    def test_infer_contract(self):
        config = self._config()
        trainer = SAMTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_sam")
            trainer.train(config, make_sam_dataloader(config), logger)
            img = torch.rand(1, 1, 56, 56)
            out = trainer.infer(config, {"images": img, "points": [[20, 20]], "labels": [1]})
            assert out["masks"].shape == (1, 3, 56, 56)
            assert (out["masks"] >= 0).all() and (out["masks"] <= 1).all()
            assert len(out["iou_pred"][0]) == 3

    def test_checkpoint_roundtrip(self):
        config = self._config()
        trainer = SAMTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_sam")
            trainer.train(config, make_sam_dataloader(config), logger)
            fresh = SAMTrainer()
            fresh.load_checkpoint(config, logger.artifact_path("model.pt").parent)
            assert fresh.model is not None
            # Fourier prompt buffer restored from the checkpoint, not re-randomized
            assert torch.equal(fresh.model.prompt_encoder.freq,
                               trainer.model.prompt_encoder.freq)
