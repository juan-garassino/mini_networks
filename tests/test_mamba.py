"""Smoke tests for NanoMamba: MambaBlock, NanoMamba model, MambaTrainer."""
import os
import tempfile

import torch
import torch.nn.functional as F

from mini_networks.models.mamba.config import MambaConfig
from mini_networks.models.mamba.model import MambaBlock, NanoMamba
from mini_networks.models.mamba.trainer import MambaTrainer, make_mamba_dataloader
from mini_networks.core.logging.logger import Logger

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


# ---------------------------------------------------------------------------
# MambaBlock unit tests
# ---------------------------------------------------------------------------

class TestMambaBlock:
    def test_output_shape(self):
        block = MambaBlock(d_model=32, d_state=8, d_conv=4, dropout=0.0)
        x = torch.randn(2, 16, 32)
        out = block(x)
        assert out.shape == (2, 16, 32)

    def test_residual(self):
        """Output should differ from input (residual adds SSM contribution)."""
        block = MambaBlock(d_model=32, d_state=8, d_conv=4, dropout=0.0)
        x = torch.randn(2, 8, 32)
        out = block(x)
        assert not torch.allclose(out, x)

    def test_no_nan(self):
        block = MambaBlock(d_model=32, d_state=8, d_conv=4)
        x = torch.randn(4, 16, 32)
        out = block(x)
        assert not torch.isnan(out).any()

    def test_variable_seq_len(self):
        """Block must handle any sequence length without shape errors."""
        block = MambaBlock(d_model=16, d_state=4, d_conv=4, dropout=0.0)
        for T in [1, 4, 16, 64]:
            x = torch.randn(1, T, 16)
            out = block(x)
            assert out.shape == (1, T, 16)

    def test_backprop(self):
        block = MambaBlock(d_model=16, d_state=4, d_conv=4, dropout=0.0)
        x = torch.randn(2, 8, 16)
        out = block(x)
        out.mean().backward()
        for p in block.parameters():
            assert p.grad is not None

    def test_causal_conv_trim(self):
        """depthwise conv padding is trimmed so output length == input length."""
        block = MambaBlock(d_model=8, d_state=4, d_conv=4, dropout=0.0)
        for T in [1, 3, 8]:
            x = torch.randn(1, T, 8)
            out = block(x)
            assert out.shape[-2] == T


# ---------------------------------------------------------------------------
# NanoMamba model tests
# ---------------------------------------------------------------------------

class TestNanoMamba:
    def _model(self, **kwargs):
        defaults = dict(vocab_size=64, d_model=32, n_layers=2,
                        d_state=8, d_conv=4, seq_len=16, dropout=0.0)
        defaults.update(kwargs)
        return NanoMamba(**defaults)

    def test_forward_shape(self):
        model = self._model()
        tokens = torch.randint(0, 64, (2, 16))
        logits, aux = model(tokens)
        assert logits.shape == (2, 16, 64)
        assert aux.item() == 0.0

    def test_no_nan(self):
        model = self._model()
        tokens = torch.randint(0, 64, (4, 16))
        logits, _ = model(tokens)
        assert not torch.isnan(logits).any()

    def test_generate_length(self):
        model = self._model()
        prompt = torch.randint(0, 64, (1, 4))
        out = model.generate(prompt, max_new_tokens=8)
        assert out.shape == (1, 12)

    def test_generate_prompt_preserved(self):
        """Prompt tokens must appear unchanged at the start of the output."""
        model = self._model()
        prompt = torch.randint(0, 64, (1, 4))
        out = model.generate(prompt.clone(), max_new_tokens=6)
        assert torch.equal(out[:, :4], prompt)

    def test_backprop(self):
        model = self._model()
        model.train()
        tokens = torch.randint(0, 64, (2, 8))
        logits, _ = model(tokens)
        targets = torch.randint(0, 64, (2, 8))
        loss = F.cross_entropy(logits.view(-1, 64), targets.view(-1))
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_deeper_model(self):
        model = self._model(n_layers=6, d_model=64)
        tokens = torch.randint(0, 64, (2, 16))
        logits, _ = model(tokens)
        assert logits.shape == (2, 16, 64)

    def test_aux_always_zero(self):
        """NanoMamba never produces auxiliary loss — it's always 0."""
        model = self._model()
        model.train()
        tokens = torch.randint(0, 64, (2, 8))
        _, aux = model(tokens)
        assert aux.item() == 0.0


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------

class TestMambaTrainer:
    def _config(self, **kwargs):
        defaults = dict(
            vocab_size=64, d_model=32, n_layers=1,
            d_state=8, d_conv=4, seq_len=16,
            fast_demo=True, data_root=DATA_ROOT, epochs=1,
        )
        defaults.update(kwargs)
        return MambaConfig(**defaults)

    def test_train_smoke(self):
        config = self._config()
        trainer = MambaTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_mamba")
            dl = make_mamba_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            assert len(metrics) > 0

    def test_evaluate(self):
        config = self._config()
        trainer = MambaTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_mamba")
            dl = make_mamba_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.evaluate(config, dl, logger)
            assert "eval_loss" in result

    def test_infer_shakespeare(self):
        config = self._config(vocab_size=128)
        trainer = MambaTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_mamba")
            dl = make_mamba_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.infer(config, {"prompt": "KING", "max_new_tokens": 8})
            assert "generated" in result
            assert isinstance(result["generated"], str)

    def test_checkpoint_saved(self):
        config = self._config()
        trainer = MambaTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_mamba")
            dl = make_mamba_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            assert logger.artifact_path("model.pt").exists()
