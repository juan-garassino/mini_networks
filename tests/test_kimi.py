"""Tests for mini Kimi K3: KDA, SiTU-GLU, LatentMoE, AttnRes, KimiLM, KimiTrainer."""
import os
import tempfile

import torch
import torch.nn.functional as F

from mini_networks.core.logging.logger import Logger
from mini_networks.models.kimi.config import KimiConfig
from mini_networks.models.kimi.model import (
    KDALayer,
    KimiLM,
    SiTUGLU,
    StableLatentMoE,
)
from mini_networks.models.kimi.trainer import KimiTrainer, make_kimi_dataloader

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


class TestKDALayer:
    def test_output_shape(self):
        layer = KDALayer(d_model=32, n_heads=2)
        x = torch.randn(2, 16, 32)
        assert layer(x).shape == (2, 16, 32)

    def test_no_nan(self):
        layer = KDALayer(d_model=32, n_heads=2)
        x = torch.randn(4, 16, 32) * 10
        assert torch.isfinite(layer(x)).all()

    def test_causality(self):
        """Perturbing a future token must not change earlier outputs."""
        layer = KDALayer(d_model=16, n_heads=2)
        layer.eval()
        x = torch.randn(1, 12, 16)
        x2 = x.clone()
        x2[:, 8:] += 1.0
        with torch.no_grad():
            o1, o2 = layer(x), layer(x2)
        assert torch.allclose(o1[:, :8], o2[:, :8], atol=1e-5)

    def test_decay_bounded(self):
        """Retention factors stay in (e^{g_min}, 1) — the K3 bounded-decay claim."""
        layer = KDALayer(d_model=16, n_heads=2, g_min=-5.0)
        x = torch.randn(2, 8, 16) * 100
        z = layer._heads(layer.alpha_up(layer.alpha_down(x)) + layer.alpha_bias)
        g = layer.g_min * torch.sigmoid(
            torch.exp(layer.log_scale).view(1, 1, 2, 1) * z
        )
        alpha = torch.exp(g)
        assert (alpha > torch.e ** -5 - 1e-6).all()
        assert (alpha <= 1.0).all()  # open interval mathematically; fp32 saturates to 1.0

    def test_backprop(self):
        layer = KDALayer(d_model=16, n_heads=2)
        x = torch.randn(2, 8, 16)
        layer(x).mean().backward()
        for p in layer.parameters():
            assert p.grad is not None


class TestSiTUGLU:
    def test_shape(self):
        m = SiTUGLU(16, 32)
        assert m(torch.randn(4, 16)).shape == (4, 16)

    def test_softcap_survives_huge_inputs(self):
        """Both GLU factors are softcapped, so extreme inputs can't overflow."""
        m = SiTUGLU(16, 32, beta1=4.0, beta2=25.0)
        out = m(torch.randn(4, 16) * 1e6)
        assert torch.isfinite(out).all()

    def test_branch_product_bounded(self):
        m = SiTUGLU(16, 32, beta1=4.0, beta2=25.0)
        x = torch.randn(64, 16) * 1e3
        g = m.w_g(x)
        gate = m.beta1 * torch.tanh(g / m.beta1) * torch.sigmoid(g)
        up = m.beta2 * torch.tanh(m.w_u(x) / m.beta2)
        assert (gate * up).abs().max() <= m.beta1 * m.beta2 + 1e-4


class TestStableLatentMoE:
    def test_shape_and_zero_aux(self):
        moe = StableLatentMoE(d_model=32, d_ff=64, latent_dim=16, num_routed=4, top_k=2)
        y, aux = moe(torch.randn(2, 8, 32))
        assert y.shape == (2, 8, 32)
        assert aux.item() == 0.0  # aux-loss-free balancing

    def test_bias_updates_only_in_train(self):
        moe = StableLatentMoE(d_model=32, d_ff=64, latent_dim=16, num_routed=4, top_k=1)
        moe.train()
        for _ in range(3):
            moe(torch.randn(4, 8, 32))
        x = torch.randn(4, 8, 32)
        after_train = moe.route_bias.clone()
        moe.eval()
        moe(x)
        assert not torch.equal(after_train, torch.zeros(4))  # moved during training
        assert torch.equal(moe.route_bias, after_train)      # frozen at eval


class TestKimiLM:
    def _model(self, **kwargs):
        defaults = dict(
            vocab_size=64, d_model=32, n_heads=2, n_layers=4, d_ff=64,
            seq_len=16, dropout=0.0, attn_res_block_size=4, latent_dim=16,
            num_routed=4, router_top_k=2, kda_decay_rank=8,
        )
        defaults.update(kwargs)
        return KimiLM(**defaults)

    def test_forward_shape(self):
        model = self._model()
        logits, aux = model(torch.randint(0, 64, (2, 16)))
        assert logits.shape == (2, 16, 64)

    def test_causality(self):
        """Full-model check: KDA + gated attention + AttnRes must all stay causal."""
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
        assert aux_train.item() > 0.0   # MTP CE rides the aux channel
        assert aux_eval.item() == 0.0

    def test_attn_res_ablation(self):
        model = self._model(use_attn_res=False)
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
        n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        assert n_with_grad > 0


class TestKimiTrainer:
    def _config(self, **kwargs):
        defaults = dict(
            vocab_size=128, d_model=32, n_heads=2, n_layers=4, d_ff=64,
            seq_len=16, latent_dim=16, num_routed=4, kda_decay_rank=8,
            fast_demo=True, data_root=DATA_ROOT, epochs=1,
        )
        defaults.update(kwargs)
        return KimiConfig(**defaults)

    def test_train_smoke(self):
        config = self._config()
        trainer = KimiTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_kimi")
            dl = make_kimi_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            assert len(logger.read_metrics()) > 0
            assert logger.artifact_path("model.pt").exists()

    def test_evaluate_and_infer(self):
        config = self._config()
        trainer = KimiTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_kimi")
            dl = make_kimi_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.evaluate(config, dl, logger)
            assert "eval_loss" in result
            gen = trainer.infer(config, {"prompt": "KING", "max_new_tokens": 8})
            assert isinstance(gen["generated"], str)

    def test_checkpoint_roundtrip(self):
        config = self._config()
        trainer = KimiTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_kimi")
            dl = make_kimi_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            fresh = KimiTrainer()
            fresh.load_checkpoint(config, logger.artifact_path("model.pt").parent)
            assert fresh.model is not None
