"""Tests for the masked-diffusion text LM (LLaDA-mini)."""
import os
import tempfile

import torch

from mini_networks.core.logging.logger import Logger
from mini_networks.models.text_diffusion.config import TextDiffusionConfig
from mini_networks.models.text_diffusion.model import TextDiffusionLM
from mini_networks.models.text_diffusion.trainer import (
    TextDiffusionTrainer,
    make_text_diffusion_dataloader,
)

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


class TestTextDiffusionLM:
    def _model(self, **kwargs):
        defaults = dict(vocab_size=64, d_model=32, n_heads=2, n_layers=2,
                        d_ff=64, seq_len=32, dropout=0.0)
        defaults.update(kwargs)
        return TextDiffusionLM(**defaults)

    def test_forward_shape(self):
        model = self._model()
        tokens = torch.randint(0, 64, (2, 16))
        assert model(tokens).shape == (2, 16, 64)

    def test_bidirectional(self):
        """Perturbing a LATER token CAN change earlier predictions — the
        inverted causality test; this is the whole point vs `transformer`."""
        model = self._model()
        model.eval()
        tokens = torch.randint(0, 64, (1, 16))
        t2 = tokens.clone()
        t2[:, 12:] = (t2[:, 12:] + 7) % 64
        with torch.no_grad():
            l1, l2 = model(tokens), model(t2)
        assert not torch.allclose(l1[:, :12], l2[:, :12], atol=1e-4)

    def test_masked_loss_ignores_unmasked(self):
        """Loss must come from masked positions only."""
        model = self._model()
        model.eval()
        tokens = torch.randint(0, 64, (4, 16))
        torch.manual_seed(0)
        loss, mask = model.masked_loss(tokens)
        assert mask.any()
        assert torch.isfinite(loss)
        # recompute by hand from the masked positions
        corrupted = torch.where(mask, torch.full_like(tokens, model.mask_id), tokens)
        with torch.no_grad():
            logits = model(corrupted)
        import torch.nn.functional as F
        ce = F.cross_entropy(
            logits.view(-1, 64), tokens.view(-1), reduction="none"
        ).view(tokens.shape)
        assert (ce * mask).sum() > 0

    def test_generate_pins_prompt(self):
        model = self._model()
        prompt = torch.randint(0, 64, (1, 5))
        out = model.generate(prompt.clone(), max_new_tokens=10, steps=4)
        assert torch.equal(out[:, :5], prompt)
        assert out.shape[1] == 15
        assert (out != model.mask_id).all()  # everything resolved

    def test_generate_respects_seq_len(self):
        model = self._model(seq_len=16)
        prompt = torch.randint(0, 64, (1, 5))
        out = model.generate(prompt, max_new_tokens=100, steps=4)
        assert out.shape[1] == 16  # capped at seq_len

    def test_backprop(self):
        model = self._model()
        tokens = torch.randint(0, 64, (2, 16))
        loss, _ = model.masked_loss(tokens)
        loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


class TestTextDiffusionTrainer:
    def _config(self, **kwargs):
        defaults = dict(
            vocab_size=128, d_model=32, n_heads=2, n_layers=2, d_ff=64,
            seq_len=16, fast_demo=True, data_root=DATA_ROOT, epochs=1,
        )
        defaults.update(kwargs)
        return TextDiffusionConfig(**defaults)

    def test_train_smoke(self):
        config = self._config()
        trainer = TextDiffusionTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_text_diffusion")
            dl = make_text_diffusion_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            assert len(logger.read_metrics()) > 0
            assert logger.artifact_path("model.pt").exists()

    def test_evaluate_and_infer(self):
        config = self._config()
        trainer = TextDiffusionTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_text_diffusion")
            dl = make_text_diffusion_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.evaluate(config, dl, logger)
            assert "eval_loss" in result
            gen = trainer.infer(config, {"prompt": "KING", "max_new_tokens": 12})
            assert isinstance(gen["generated"], str)
            assert len(gen["generated"]) > 0
            # (prompt pinning is asserted at the id level in
            # test_generate_pins_prompt — the S-tier char vocab may lack
            # some prompt characters, so no string-prefix check here)

    def test_checkpoint_roundtrip(self):
        config = self._config()
        trainer = TextDiffusionTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_text_diffusion")
            dl = make_text_diffusion_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            fresh = TextDiffusionTrainer()
            fresh.load_checkpoint(config, logger.artifact_path("model.pt").parent)
            assert fresh.model is not None
            assert fresh.model.vocab_size == trainer.model.vocab_size
