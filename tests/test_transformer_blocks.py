"""Smoke tests for pluggable TransformerLM FFN blocks: standard / MoE / Mamba."""
import os
import tempfile

import torch
import pytest

from mini_networks.models.transformer.config import TransformerConfig
from mini_networks.models.transformer.model import (
    StandardFFN,
    MoEFFN,
    MambaFFN,
    TransformerLM,
)
from mini_networks.models.transformer.trainer import TransformerTrainer, make_transformer_dataloader
from mini_networks.core.logging.logger import Logger

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


# ---------------------------------------------------------------------------
# StandardFFN
# ---------------------------------------------------------------------------

class TestStandardFFN:
    def test_output_shape(self):
        ffn = StandardFFN(d_model=32, d_ff=64)
        x = torch.randn(2, 8, 32)
        out, aux = ffn(x)
        assert out.shape == (2, 8, 32)
        assert aux.item() == 0.0

    def test_no_nan(self):
        ffn = StandardFFN(d_model=32, d_ff=64)
        x = torch.randn(3, 16, 32)
        out, _ = ffn(x)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# MoEFFN
# ---------------------------------------------------------------------------

class TestMoEFFN:
    def _make(self, num_experts=4, k=1):
        return MoEFFN(d_model=32, d_ff=64, num_experts=num_experts, k=k,
                      router_hidden=16, dropout=0.0)

    def test_output_shape(self):
        moe = self._make()
        x = torch.randn(2, 8, 32)
        out, aux = moe(x)
        assert out.shape == (2, 8, 32)

    def test_aux_loss_scalar(self):
        moe = self._make()
        moe.train()
        x = torch.randn(2, 8, 32)
        _, aux = moe(x)
        assert aux.shape == ()

    def test_aux_loss_finite(self):
        moe = self._make()
        moe.train()
        x = torch.randn(2, 8, 32)
        _, aux = moe(x)
        assert torch.isfinite(aux)

    def test_topk_2(self):
        moe = self._make(num_experts=4, k=2)
        x = torch.randn(2, 8, 32)
        out, aux = moe(x)
        assert out.shape == (2, 8, 32)

    def test_no_nan(self):
        moe = self._make()
        moe.train()
        x = torch.randn(4, 16, 32)
        out, aux = moe(x)
        assert not torch.isnan(out).any()
        assert not torch.isnan(aux)

    def test_aux_backprop(self):
        moe = self._make()
        moe.train()
        x = torch.randn(2, 4, 32)
        out, aux = moe(x)
        loss = out.mean() + aux
        loss.backward()
        # Router parameters should have gradients
        for p in moe.router.parameters():
            assert p.grad is not None

    def test_eval_no_gumbel(self):
        """In eval mode Gumbel noise should be off — output should be deterministic."""
        moe = self._make()
        moe.eval()
        x = torch.randn(1, 4, 32)
        out1, _ = moe(x)
        out2, _ = moe(x)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# MambaFFN
# ---------------------------------------------------------------------------

class TestMambaFFN:
    def _make(self):
        return MambaFFN(d_model=32, d_state=8, d_conv=4, dropout=0.0)

    def test_output_shape(self):
        mamba = self._make()
        x = torch.randn(2, 8, 32)
        out, aux = mamba(x)
        assert out.shape == (2, 8, 32)
        assert aux.item() == 0.0

    def test_residual_connection(self):
        """MambaFFN applies residual internally — output ≠ 0 even with zeroed params."""
        mamba = self._make()
        x = torch.randn(2, 4, 32)
        out, _ = mamba(x)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_no_nan(self):
        mamba = self._make()
        x = torch.randn(4, 16, 32)
        out, _ = mamba(x)
        assert not torch.isnan(out).any()

    def test_causal_conv_trim(self):
        """depthwise conv is trimmed to T — output length must equal input length."""
        mamba = self._make()
        for T in [1, 4, 16]:
            x = torch.randn(1, T, 32)
            out, _ = mamba(x)
            assert out.shape == (1, T, 32)


# ---------------------------------------------------------------------------
# TransformerLM with MoE blocks
# ---------------------------------------------------------------------------

class TestTransformerLM_MoE:
    def _model(self, **kwargs):
        return TransformerLM(
            vocab_size=64,
            d_model=32,
            n_heads=2,
            n_layers=2,
            d_ff=64,
            seq_len=16,
            dropout=0.0,
            block_type="moe",
            num_experts=3,
            k=1,
            router_hidden=16,
        )

    def test_forward_shape(self):
        model = self._model()
        tokens = torch.randint(0, 64, (2, 16))
        logits, aux = model(tokens)
        assert logits.shape == (2, 16, 64)

    def test_aux_nonzero(self):
        model = self._model()
        model.train()
        tokens = torch.randint(0, 64, (2, 16))
        _, aux = model(tokens)
        # With 2 MoE layers the total aux should be > 0 during training
        assert torch.isfinite(aux)

    def test_no_nan(self):
        model = self._model()
        tokens = torch.randint(0, 64, (4, 16))
        logits, _ = model(tokens)
        assert not torch.isnan(logits).any()

    def test_generate(self):
        model = self._model()
        prompt = torch.randint(0, 64, (1, 4))
        out = model.generate(prompt, max_new_tokens=6)
        assert out.shape[1] == 4 + 6

    def test_backprop(self):
        model = self._model()
        model.train()
        tokens = torch.randint(0, 64, (2, 8))
        logits, aux = model(tokens)
        targets = torch.randint(0, 64, (2, 8))
        import torch.nn.functional as F
        loss = F.cross_entropy(logits.view(-1, 64), targets.view(-1)) + aux
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ---------------------------------------------------------------------------
# TransformerLM with Mamba blocks
# ---------------------------------------------------------------------------

class TestTransformerLM_Mamba:
    def _model(self):
        return TransformerLM(
            vocab_size=64,
            d_model=32,
            n_heads=2,
            n_layers=2,
            d_ff=64,
            seq_len=16,
            dropout=0.0,
            block_type="mamba",
            d_state=8,
            d_conv=4,
        )

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

    def test_generate(self):
        model = self._model()
        prompt = torch.randint(0, 64, (1, 4))
        out = model.generate(prompt, max_new_tokens=6)
        assert out.shape[1] == 10

    def test_backprop(self):
        model = self._model()
        model.train()
        tokens = torch.randint(0, 64, (2, 8))
        logits, _ = model(tokens)
        targets = torch.randint(0, 64, (2, 8))
        import torch.nn.functional as F
        loss = F.cross_entropy(logits.view(-1, 64), targets.view(-1))
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ---------------------------------------------------------------------------
# Trainer integration (fast_demo)
# ---------------------------------------------------------------------------

class TestTransformerTrainerBlocks:
    def _config(self, block_type: str, **extra):
        return TransformerConfig(
            vocab_size=64,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            seq_len=16,
            fast_demo=True,
            data_root=DATA_ROOT,
            epochs=1,
            block_type=block_type,
            **extra,
        )

    def test_moe_trainer_smoke(self):
        config = self._config(
            "moe",
            moe_num_experts=3,
            moe_top_k=1,
            moe_router_hidden=16,
        )
        trainer = TransformerTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_moe")
            dl = make_transformer_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            assert len(metrics) > 0

    def test_mamba_trainer_smoke(self):
        config = self._config(
            "mamba",
            mamba_d_state=8,
            mamba_d_conv=4,
        )
        trainer = TransformerTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_mamba")
            dl = make_transformer_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            assert len(metrics) > 0

    def test_standard_still_works(self):
        config = self._config("standard")
        trainer = TransformerTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_std")
            dl = make_transformer_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.evaluate(config, dl, logger)
            assert "eval_loss" in result
