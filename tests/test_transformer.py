"""Transformer: tokenizer, model forward, trainer 1-step smoke test."""
import os
import tempfile

import torch

from mini_networks.models.transformer.config import TransformerConfig
from mini_networks.models.transformer.model import TransformerLM
from mini_networks.models.transformer.tokenizer import BPETokenizer, CharTokenizer
from mini_networks.models.transformer.trainer import TransformerTrainer, make_transformer_dataloader
from mini_networks.core.logging.logger import Logger

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


class TestCharTokenizer:
    def test_round_trip(self):
        text = "hello world"
        tok = CharTokenizer.from_text(text)
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_vocab_size(self):
        tok = CharTokenizer.from_text("abcdef")
        assert tok.vocab_size >= 6

    def test_unknown_char(self):
        tok = CharTokenizer.from_text("abc")
        ids = tok.encode("xyz")
        # Unknown chars map to PAD (0)
        assert all(i == 0 for i in ids)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = CharTokenizer.from_text("hello world")
            path = os.path.join(tmpdir, "tokenizer.json")
            tok.save(path)
            tok2 = CharTokenizer.load(path)
            assert tok2.vocab_size == tok.vocab_size
            assert tok2.encode("hello") == tok.encode("hello")


class TestTransformerLM:
    def test_forward_shape(self):
        model = TransformerLM(vocab_size=32, d_model=32, n_heads=2, n_layers=1, d_ff=64, seq_len=16)
        tokens = torch.randint(0, 32, (2, 16))
        logits, aux = model(tokens)
        assert logits.shape == (2, 16, 32)
        assert aux.item() == 0.0  # standard FFN has no aux loss

    def test_generate(self):
        model = TransformerLM(vocab_size=32, d_model=32, n_heads=2, n_layers=1, d_ff=64, seq_len=16)
        prompt = torch.randint(0, 32, (1, 4))
        out = model.generate(prompt, max_new_tokens=8)
        assert out.shape[1] == 4 + 8

    def test_no_nan(self):
        model = TransformerLM(vocab_size=64, d_model=32, n_heads=2, n_layers=2, d_ff=64, seq_len=16)
        tokens = torch.randint(0, 64, (4, 16))
        logits, _ = model(tokens)
        assert not torch.isnan(logits).any()


class TestBPETokenizer:
    CORPUS = "hello world hello python hello neural networks are great great great"

    def test_train_no_error(self):
        tok = BPETokenizer()
        tok.train(self.CORPUS, vocab_size=300)
        assert tok.vocab_size >= 256

    def test_encode_decode_round_trip(self):
        tok = BPETokenizer()
        tok.train(self.CORPUS, vocab_size=300)
        text = "hello world"
        decoded = tok.decode(tok.encode(text))
        assert decoded == text

    def test_encode_reduces_length(self):
        """BPE should merge common pairs, reducing sequence length."""
        tok = BPETokenizer()
        tok.train(self.CORPUS, vocab_size=320)
        ids_bpe = tok.encode("hello world hello")
        ids_bytes = list("hello world hello".encode("utf-8"))
        # After merges, BPE ids should be <= raw bytes
        assert len(ids_bpe) <= len(ids_bytes)

    def test_vocab_size_respects_target(self):
        tok = BPETokenizer()
        tok.train(self.CORPUS, vocab_size=280)
        assert tok.vocab_size <= 280

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = BPETokenizer()
            tok.train(self.CORPUS, vocab_size=300)
            path = os.path.join(tmpdir, "bpe.json")
            tok.save(path)
            tok2 = BPETokenizer.load(path)
            assert tok2.vocab_size == tok.vocab_size
            assert tok2.encode("hello") == tok.encode("hello")

    def test_empty_text_encodes_to_empty(self):
        tok = BPETokenizer()
        tok.train(self.CORPUS, vocab_size=280)
        assert tok.encode("") == []

    def test_unicode_round_trip(self):
        tok = BPETokenizer()
        tok.train("hello world " * 20, vocab_size=280)
        text = "caf\u00e9"  # café
        assert tok.decode(tok.encode(text)) == text

    def test_merges_count(self):
        tok = BPETokenizer()
        tok.train(self.CORPUS, vocab_size=310)
        # Number of merges should be vocab_size - 256 at most
        assert len(tok.merges) <= 310 - 256


class TestBPETransformerTrainer:
    def test_train_with_bpe(self):
        config = TransformerConfig(
            d_model=32, n_heads=2, n_layers=1, d_ff=64, seq_len=16,
            fast_demo=True, data_root=DATA_ROOT, epochs=1,
            tokenizer_type="bpe", bpe_vocab_size=300,
        )
        trainer = TransformerTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_bpe")
            dl = make_transformer_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            assert len(metrics) > 0

    def test_bpe_infer_returns_string(self):
        config = TransformerConfig(
            d_model=32, n_heads=2, n_layers=1, d_ff=64, seq_len=16,
            fast_demo=True, data_root=DATA_ROOT, epochs=1,
            tokenizer_type="bpe", bpe_vocab_size=300,
        )
        trainer = TransformerTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_bpe_infer")
            dl = make_transformer_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.infer(config, {"prompt": "KING", "max_new_tokens": 8})
            assert "generated" in result
            assert isinstance(result["generated"], str)


class TestTransformerTrainer:
    def test_train_smoke(self):
        config = TransformerConfig(
            vocab_size=64, d_model=32, n_heads=2, n_layers=1, d_ff=64, seq_len=16,
            fast_demo=True, data_root=DATA_ROOT, epochs=1,
        )
        trainer = TransformerTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test")
            dl = make_transformer_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            assert len(metrics) > 0

    def test_evaluate(self):
        config = TransformerConfig(
            vocab_size=64, d_model=32, n_heads=2, n_layers=1, d_ff=64, seq_len=16,
            fast_demo=True, data_root=DATA_ROOT, epochs=1,
        )
        trainer = TransformerTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test")
            dl = make_transformer_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.evaluate(config, dl, logger)
            assert "eval_loss" in result

    def test_infer_shakespeare(self):
        config = TransformerConfig(
            vocab_size=128, d_model=32, n_heads=2, n_layers=1, d_ff=64, seq_len=16,
            fast_demo=True, data_root=DATA_ROOT, epochs=1,
        )
        trainer = TransformerTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test")
            dl = make_transformer_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.infer(config, {"prompt": "KING", "max_new_tokens": 8})
            # tokenizer is set from Shakespeare → returns text
            assert "generated" in result
            assert isinstance(result["generated"], str)
