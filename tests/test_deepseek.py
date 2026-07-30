"""Tests for mini DeepSeek V4: compressor, CSA/HCA, mHC, MoE, DeepseekLM, trainer."""
import os
import tempfile

import torch
import torch.nn.functional as F

from mini_networks.core.logging.logger import Logger
from mini_networks.models.deepseek.config import DeepseekConfig
from mini_networks.models.deepseek.model import (
    CompressedAttention,
    DeepseekLM,
    DeepSeekMoEFFN,
    MHCMixer,
    TokenCompressor,
    sinkhorn,
)
from mini_networks.models.deepseek.trainer import DeepseekTrainer, make_deepseek_dataloader

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


class TestSinkhorn:
    def test_doubly_stochastic(self):
        m = sinkhorn(torch.rand(2, 3, 4, 4) + 0.1, iters=20)
        assert torch.allclose(m.sum(dim=-1), torch.ones(2, 3, 4), atol=1e-3)
        assert torch.allclose(m.sum(dim=-2), torch.ones(2, 3, 4), atol=1e-3)
        assert (m >= 0).all()


class TestMHCMixer:
    def test_constraints(self):
        mixer = MHCMixer(d_model=16, n=2, sinkhorn_iters=20)
        X = torch.randn(2, 8, 2, 16)
        A, B, C = mixer(X)
        assert ((A > 0) & (A < 1)).all()          # sigmoid-bounded input map
        assert ((C > 0) & (C < 2)).all()          # 2*sigmoid-bounded output map
        assert torch.allclose(B.sum(dim=-1), torch.ones(2, 8, 2), atol=1e-3)
        assert torch.allclose(B.sum(dim=-2), torch.ones(2, 8, 2), atol=1e-3)


class TestTokenCompressor:
    def test_shape(self):
        comp = TokenCompressor(d_model=16, c=8, m=4)
        out = comp(torch.randn(2, 16, 16))
        assert out.shape == (2, 4, 8)

    def test_short_sequence_empty(self):
        comp = TokenCompressor(d_model=16, c=8, m=8)
        out = comp(torch.randn(2, 5, 16))
        assert out.shape == (2, 0, 8)


class TestCompressedAttention:
    def _csa(self):
        return CompressedAttention(
            d_model=32, c=16, n_heads=2, m=4, top_k=2, n_win=4, rope_dims=8
        )

    def _hca(self):
        return CompressedAttention(
            d_model=32, c=16, n_heads=2, m=8, top_k=None, n_win=4, rope_dims=8
        )

    def test_shapes(self):
        x = torch.randn(2, 16, 32)
        assert self._csa()(x).shape == (2, 16, 32)
        assert self._hca()(x).shape == (2, 16, 32)

    def test_causality_csa(self):
        """The classic compressed-KV bug: future tokens leaking through pooled entries."""
        attn = self._csa()
        attn.eval()
        x = torch.randn(1, 16, 32)
        x2 = x.clone()
        x2[:, 12:] += 1.0
        with torch.no_grad():
            o1, o2 = attn(x), attn(x2)
        assert torch.allclose(o1[:, :12], o2[:, :12], atol=1e-5)

    def test_causality_hca(self):
        attn = self._hca()
        attn.eval()
        x = torch.randn(1, 24, 32)
        x2 = x.clone()
        x2[:, 17:] += 1.0
        with torch.no_grad():
            o1, o2 = attn(x), attn(x2)
        assert torch.allclose(o1[:, :17], o2[:, :17], atol=1e-5)

    def test_tiny_sequence(self):
        """T < m: no compressed entries — window + sink must carry the layer."""
        attn = self._csa()
        out = attn(torch.randn(1, 2, 32))
        assert out.shape == (1, 2, 32)
        assert torch.isfinite(out).all()


class TestDeepSeekMoEFFN:
    def test_routed_shape_and_seq_balance_aux(self):
        moe = DeepSeekMoEFFN(d_model=32, d_ff=64, num_experts=4, top_k=2)
        moe.train()
        y, aux = moe(torch.randn(2, 8, 32))
        assert y.shape == (2, 8, 32)
        assert aux.item() > 0.0  # sequence-wise balance loss (train only)
        moe.eval()
        _, aux_eval = moe(torch.randn(2, 8, 32))
        assert aux_eval.item() == 0.0

    def test_hash_route(self):
        moe = DeepSeekMoEFFN(d_model=32, d_ff=64, num_experts=4, top_k=2, hash_route=True)
        tokens = torch.randint(0, 64, (2, 8))
        y, aux = moe(torch.randn(2, 8, 32), token_ids=tokens)
        assert y.shape == (2, 8, 32)
        assert aux.item() == 0.0  # hash routing has no balance losses


class TestDeepseekLM:
    def _model(self, **kwargs):
        defaults = dict(
            vocab_size=64, d_model=32, n_heads=2, n_layers=2, d_ff=64,
            seq_len=16, dropout=0.0, csa_m=4, csa_top_k=2, hca_m=8,
            hca_every=2, kv_dim=16, n_win=4, rope_dims=8, n_hc=2,
            ds_num_experts=4, ds_top_k=2,
        )
        defaults.update(kwargs)
        return DeepseekLM(**defaults)

    def test_forward_shape(self):
        model = self._model()
        logits, _ = model(torch.randint(0, 64, (2, 16)))
        assert logits.shape == (2, 16, 64)

    def test_causality(self):
        """Full-model check: compressor + window + mHC + MoE must all stay causal."""
        model = self._model()
        model.eval()
        tokens = torch.randint(0, 64, (1, 16))
        t2 = tokens.clone()
        t2[:, 10:] = (t2[:, 10:] + 7) % 64
        with torch.no_grad():
            l1, _ = model(tokens)
            l2, _ = model(t2)
        assert torch.allclose(l1[:, :10], l2[:, :10], atol=1e-4)

    def test_mtp_aux_in_train_only(self):
        model = self._model()
        tokens = torch.randint(0, 64, (2, 16))
        model.train()
        _, aux_train = model(tokens)
        model.eval()
        with torch.no_grad():
            _, aux_eval = model(tokens)
        assert aux_train.item() > 0.0
        assert aux_eval.item() == 0.0

    def test_mhc_ablation(self):
        model = self._model(use_mhc=False)
        logits, _ = model(torch.randint(0, 64, (2, 16)))
        assert logits.shape == (2, 16, 64)

    def test_generate(self):
        model = self._model()
        prompt = torch.randint(0, 64, (1, 4))
        out = model.generate(prompt.clone(), max_new_tokens=6)
        assert out.shape == (1, 10)
        assert torch.equal(out[:, :4], prompt)

    def test_backprop(self):
        model = self._model()
        model.train()
        tokens = torch.randint(0, 64, (2, 16))
        logits, aux = model(tokens)
        loss = F.cross_entropy(logits.view(-1, 64), tokens.view(-1)) + aux
        loss.backward()
        assert sum(1 for p in model.parameters() if p.grad is not None) > 0


class TestDeepseekTrainer:
    def _config(self, **kwargs):
        defaults = dict(
            vocab_size=128, d_model=32, n_heads=2, n_layers=2, d_ff=64,
            seq_len=16, csa_m=4, csa_top_k=2, hca_m=8, hca_every=2,
            kv_dim=16, n_win=4, rope_dims=8, ds_num_experts=4,
            fast_demo=True, data_root=DATA_ROOT, epochs=1,
        )
        defaults.update(kwargs)
        return DeepseekConfig(**defaults)

    def test_train_smoke(self):
        config = self._config()
        trainer = DeepseekTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_deepseek")
            dl = make_deepseek_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            assert len(logger.read_metrics()) > 0
            assert logger.artifact_path("model.pt").exists()

    def test_evaluate_and_infer(self):
        config = self._config()
        trainer = DeepseekTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_deepseek")
            dl = make_deepseek_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.evaluate(config, dl, logger)
            assert "eval_loss" in result
            gen = trainer.infer(config, {"prompt": "KING", "max_new_tokens": 8})
            assert isinstance(gen["generated"], str)

    def test_checkpoint_roundtrip(self):
        config = self._config()
        trainer = DeepseekTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_deepseek")
            dl = make_deepseek_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            fresh = DeepseekTrainer()
            fresh.load_checkpoint(config, logger.artifact_path("model.pt").parent)
            assert fresh.model is not None
