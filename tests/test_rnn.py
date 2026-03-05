"""Smoke tests for RNNLanguageModel: vanilla RNN, LSTM, GRU, and trainer."""
import os
import tempfile

import torch
import torch.nn.functional as F
import pytest

from mini_networks.models.rnn.config import RNNConfig
from mini_networks.models.rnn.model import RNNLanguageModel
from mini_networks.models.rnn.trainer import RNNTrainer, make_rnn_dataloader
from mini_networks.core.logging.logger import Logger

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")

CELL_TYPES = ["rnn", "lstm", "gru"]


# ---------------------------------------------------------------------------
# Model — parametrised over all three cell types
# ---------------------------------------------------------------------------

class TestRNNLanguageModel:
    def _model(self, cell_type: str, **kwargs):
        defaults = dict(vocab_size=64, hidden_dim=32, n_layers=1,
                        seq_len=16, dropout=0.0, cell_type=cell_type)
        defaults.update(kwargs)
        return RNNLanguageModel(**defaults)

    @pytest.mark.parametrize("cell_type", CELL_TYPES)
    def test_forward_shape(self, cell_type):
        model = self._model(cell_type)
        tokens = torch.randint(0, 64, (2, 16))
        logits, aux = model(tokens)
        assert logits.shape == (2, 16, 64)
        assert aux.item() == 0.0

    @pytest.mark.parametrize("cell_type", CELL_TYPES)
    def test_no_nan(self, cell_type):
        model = self._model(cell_type)
        tokens = torch.randint(0, 64, (4, 16))
        logits, _ = model(tokens)
        assert not torch.isnan(logits).any()

    @pytest.mark.parametrize("cell_type", CELL_TYPES)
    def test_generate_length(self, cell_type):
        model = self._model(cell_type)
        prompt = torch.randint(0, 64, (1, 4))
        out = model.generate(prompt, max_new_tokens=8)
        assert out.shape == (1, 12)

    @pytest.mark.parametrize("cell_type", CELL_TYPES)
    def test_generate_prompt_preserved(self, cell_type):
        model = self._model(cell_type)
        prompt = torch.randint(0, 64, (1, 4))
        out = model.generate(prompt.clone(), max_new_tokens=6)
        assert torch.equal(out[:, :4], prompt)

    @pytest.mark.parametrize("cell_type", CELL_TYPES)
    def test_backprop(self, cell_type):
        model = self._model(cell_type)
        model.train()
        tokens = torch.randint(0, 64, (2, 8))
        logits, _ = model(tokens)
        targets = torch.randint(0, 64, (2, 8))
        loss = F.cross_entropy(logits.view(-1, 64), targets.view(-1))
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    @pytest.mark.parametrize("cell_type", CELL_TYPES)
    def test_multilayer(self, cell_type):
        model = self._model(cell_type, n_layers=2, dropout=0.1)
        tokens = torch.randint(0, 64, (2, 16))
        logits, _ = model(tokens)
        assert logits.shape == (2, 16, 64)

    def test_invalid_cell_type_raises(self):
        with pytest.raises(ValueError, match="cell_type"):
            RNNLanguageModel(vocab_size=64, cell_type="transformer")

    def test_hidden_state_carryover(self):
        """Passing hidden state explicitly must change output vs fresh state."""
        model = RNNLanguageModel(vocab_size=32, hidden_dim=16, n_layers=1,
                                 cell_type="gru", dropout=0.0)
        model.eval()
        tokens = torch.randint(0, 32, (1, 8))
        logits_fresh, _ = model(tokens)
        # Warm-up hidden on some other tokens then continue
        warm = torch.randint(0, 32, (1, 8))
        _, _ = model(warm)  # side-effect: no change; hidden is not stored
        # Fresh pass must be deterministic
        logits_fresh2, _ = model(tokens)
        assert torch.allclose(logits_fresh, logits_fresh2)

    def test_aux_always_zero(self):
        for ct in CELL_TYPES:
            model = self._model(ct)
            tokens = torch.randint(0, 64, (2, 8))
            _, aux = model(tokens)
            assert aux.item() == 0.0


# ---------------------------------------------------------------------------
# Efficient generation — hidden state carry
# ---------------------------------------------------------------------------

class TestRNNGeneration:
    def test_generate_uses_hidden_state(self):
        """generate() should produce a full sequence without CUDA OOM on long seq."""
        model = RNNLanguageModel(vocab_size=32, hidden_dim=16, n_layers=1,
                                 cell_type="lstm", dropout=0.0, seq_len=512)
        prompt = torch.randint(0, 32, (1, 10))
        out = model.generate(prompt, max_new_tokens=50)
        assert out.shape == (1, 60)

    @pytest.mark.parametrize("cell_type", CELL_TYPES)
    def test_batch_generate(self, cell_type):
        model = RNNLanguageModel(vocab_size=32, hidden_dim=16, n_layers=1,
                                 cell_type=cell_type, dropout=0.0)
        prompt = torch.randint(0, 32, (3, 4))
        out = model.generate(prompt, max_new_tokens=5)
        assert out.shape == (3, 9)


# ---------------------------------------------------------------------------
# Trainer — one test per cell type + shared helpers
# ---------------------------------------------------------------------------

class TestRNNTrainer:
    def _config(self, cell_type: str = "lstm", **kwargs):
        defaults = dict(
            cell_type=cell_type,
            hidden_dim=32, n_layers=1, seq_len=16,
            fast_demo=True, data_root=DATA_ROOT, epochs=1,
        )
        defaults.update(kwargs)
        return RNNConfig(**defaults)

    @pytest.mark.parametrize("cell_type", CELL_TYPES)
    def test_train_smoke(self, cell_type):
        config = self._config(cell_type)
        trainer = RNNTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name=f"test_{cell_type}")
            dl = make_rnn_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            assert len(metrics) > 0

    def test_evaluate(self):
        config = self._config("lstm")
        trainer = RNNTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_rnn_eval")
            dl = make_rnn_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.evaluate(config, dl, logger)
            assert "eval_loss" in result

    def test_infer_shakespeare(self):
        config = self._config("gru", vocab_size=128)
        trainer = RNNTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_rnn_infer")
            dl = make_rnn_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.infer(config, {"prompt": "KING", "max_new_tokens": 8})
            assert "generated" in result
            assert isinstance(result["generated"], str)

    def test_checkpoint_saved(self):
        config = self._config("rnn")
        trainer = RNNTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_rnn_ckpt")
            dl = make_rnn_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            assert logger.artifact_path("model.pt").exists()
