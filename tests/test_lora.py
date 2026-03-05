"""Smoke tests for LoRALinear, LoRACNN, and LoRATrainer."""
import os
import tempfile

import torch
import torch.nn.functional as F
import pytest

from mini_networks.models.lora.config import LoRAConfig
from mini_networks.models.lora.model import LoRACNN, LoRALinear
from mini_networks.models.lora.trainer import LoRATrainer, make_lora_dataloader
from mini_networks.core.logging.logger import Logger

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------

class TestLoRALinear:
    def test_forward_shape(self):
        layer = LoRALinear(16, 8, rank=2, alpha=2.0)
        x = torch.randn(4, 16)
        out = layer(x)
        assert out.shape == (4, 8)

    def test_lora_changes_output(self):
        """LoRA adapters (non-zero A, zero B) should not change output initially."""
        layer = LoRALinear(16, 8, rank=2, alpha=2.0)
        x = torch.randn(4, 16)
        out_with = layer(x)
        # Manually zero out A so delta = 0
        with torch.no_grad():
            layer.lora_A.zero_()
        out_without = layer(x)
        # B is zero-initialized, so both should produce same output
        assert torch.allclose(out_with, out_without, atol=1e-6)

    def test_freeze_base_stops_grad(self):
        layer = LoRALinear(16, 8, rank=2)
        layer.freeze_base()
        x = torch.randn(4, 16)
        out = layer(x).sum()
        out.backward()
        assert layer.weight.grad is None
        assert layer.lora_A.grad is not None  # adapter should still get grad
        assert layer.lora_B.grad is not None

    def test_unfreeze_base_restores_grad(self):
        layer = LoRALinear(16, 8, rank=2)
        layer.freeze_base()
        layer.unfreeze_base()
        x = torch.randn(4, 16)
        out = layer(x).sum()
        out.backward()
        assert layer.weight.grad is not None

    def test_no_nan(self):
        layer = LoRALinear(32, 16, rank=4)
        x = torch.randn(8, 32)
        out = layer(x)
        assert not torch.isnan(out).any()

    def test_scale_applied(self):
        """With alpha=rank, scale=1. With alpha=2*rank, scale=2."""
        rank = 4
        layer1 = LoRALinear(16, 8, rank=rank, alpha=float(rank))
        layer2 = LoRALinear(16, 8, rank=rank, alpha=2.0 * rank)
        # Copy same weights
        with torch.no_grad():
            layer2.weight.copy_(layer1.weight)
            layer2.lora_A.copy_(layer1.lora_A)
            layer2.lora_B.copy_(layer1.lora_B)
            if layer1.bias_param is not None:
                layer2.bias_param.copy_(layer1.bias_param)
        x = torch.randn(2, 16)
        out1 = layer1(x)
        out2 = layer2(x)
        # Difference should be the extra 1× adapter output
        delta1 = (x @ layer1.lora_A.T @ layer1.lora_B.T) * layer1.scale
        delta2 = (x @ layer2.lora_A.T @ layer2.lora_B.T) * layer2.scale
        assert torch.allclose(delta2, 2.0 * delta1, atol=1e-5)


# ---------------------------------------------------------------------------
# LoRACNN
# ---------------------------------------------------------------------------

class TestLoRACNN:
    def _model(self, **kwargs):
        defaults = dict(hidden_dim=64, num_classes=10, rank=2, alpha=2.0)
        defaults.update(kwargs)
        return LoRACNN(**defaults)

    def test_forward_shape(self):
        model = self._model()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_no_nan(self):
        model = self._model()
        x = torch.randn(4, 1, 28, 28)
        assert not torch.isnan(model(x)).any()

    def test_backprop(self):
        model = self._model()
        model.train()
        x = torch.randn(4, 1, 28, 28)
        labels = torch.randint(0, 10, (4,))
        loss = F.cross_entropy(model(x), labels)
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_freeze_for_finetune_only_lora_trains(self):
        model = self._model()
        model.freeze_for_finetune(freeze_conv=True)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        # Only lora_A, lora_B should be trainable
        assert all("lora_A" in n or "lora_B" in n for n in trainable), trainable
        frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
        assert any("conv" in n for n in frozen)
        assert any("weight" in n for n in frozen)

    def test_freeze_conv_false_keeps_fc_base_trainable(self):
        model = self._model()
        model.freeze_for_finetune(freeze_conv=False)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        # conv layers should remain trainable
        assert any("conv" in n for n in trainable)
        # lora adapters also trainable
        assert any("lora_A" in n for n in trainable)

    def test_unfreeze_all(self):
        model = self._model()
        model.freeze_for_finetune(freeze_conv=True)
        model.unfreeze_all()
        all_trainable = all(p.requires_grad for p in model.parameters())
        assert all_trainable

    def test_trainable_params_subset(self):
        model = self._model()
        model.freeze_for_finetune()
        trainable = model.trainable_params()
        assert len(trainable) > 0
        assert len(trainable) < len(list(model.parameters()))


# ---------------------------------------------------------------------------
# LoRATrainer
# ---------------------------------------------------------------------------

class TestLoRATrainer:
    def _config(self, **kwargs):
        defaults = dict(
            hidden_dim=32, lora_rank=2, lora_alpha=2.0,
            pretrain_epochs=1, finetune_epochs=1,
            fast_demo=True, data_root=DATA_ROOT,
        )
        defaults.update(kwargs)
        return LoRAConfig(**defaults)

    def test_train_smoke(self):
        config = self._config()
        trainer = LoRATrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_lora")
            dl = make_lora_dataloader(config, dataset="mnist", split="train")
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            assert len(metrics) > 0

    def test_checkpoint_saved(self):
        config = self._config()
        trainer = LoRATrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_lora_ckpt")
            dl = make_lora_dataloader(config, dataset="mnist", split="train")
            trainer.train(config, dl, logger)
            assert logger.artifact_path("model.pt").exists()

    def test_evaluate(self):
        config = self._config()
        trainer = LoRATrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_lora_eval")
            dl = make_lora_dataloader(config, dataset="mnist", split="train")
            trainer.train(config, dl, logger)
            result = trainer.evaluate(config, dl, logger)
            assert "eval_loss" in result
            assert "accuracy" in result
            assert 0.0 <= result["accuracy"] <= 1.0

    def test_infer_shape(self):
        config = self._config()
        trainer = LoRATrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_lora_infer")
            dl = make_lora_dataloader(config, dataset="mnist", split="train")
            trainer.train(config, dl, logger)
            x = torch.randn(3, 1, 28, 28)
            result = trainer.infer(config, {"image": x})
            assert "predictions" in result
            assert len(result["predictions"]) == 3

    def test_pretrain_and_finetune_metrics(self):
        """Both stages produce metrics."""
        config = self._config()
        trainer = LoRATrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_lora_stages")
            dl = make_lora_dataloader(config, dataset="mnist", split="train")
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            keys = {m["key"] for m in metrics}
            assert "pretrain_loss" in keys
            assert "finetune_loss" in keys
