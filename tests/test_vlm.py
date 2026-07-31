"""Tests for the mini-VLM composition: QA encoding, loss masking, image path."""
import os
import tempfile

import torch

from mini_networks.compositions.vlm import (
    QA_TEMPLATES,
    VLM,
    VLMConfig,
    VLMNet,
    _QADataset,
    build_tokenizer,
)
from mini_networks.core.logging.logger import Logger

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


class _FakeMNIST:
    def __len__(self):
        return 64

    def __getitem__(self, idx):
        return torch.zeros(1, 28, 28), idx % 10


def _dataset(**kwargs):
    tok = build_tokenizer()
    defaults = dict(base=_FakeMNIST(), tokenizer=tok, max_len=40,
                    pad_id=len(tok.stoi))
    defaults.update(kwargs)
    return tok, _QADataset(**defaults)


class TestQAEncoding:
    def test_deterministic(self):
        _, a = _dataset()
        _, b = _dataset()
        ia, ta, la, ya = a[3]
        ib, tb, lb, yb = b[3]
        assert torch.equal(ta, tb) and torch.equal(la, lb) and ya == yb

    def test_answer_span_masking(self):
        """Labels must be -1 everywhere except targets inside ' <answer>.'."""
        tok, ds = _dataset()
        q, a_fn = QA_TEMPLATES[0]
        tokens, labels = ds.encode_qa(q, "three")
        prompt_len = len(tok.encode(f"Q: {q} A:"))
        answer_len = len(tok.encode(" three."))
        active = (labels != -1).nonzero().flatten().tolist()
        assert active == list(range(prompt_len - 1, prompt_len - 1 + answer_len))
        # the supervised targets decode back to the answer text
        target_ids = [int(labels[i]) for i in active]
        assert tok.decode(target_ids) == " three."

    def test_all_answers_encodable(self):
        tok = build_tokenizer()
        for q, a_fn in QA_TEMPLATES:
            for y in range(10):
                ids = tok.encode(f"Q: {q} A: {a_fn(y)}.")
                assert ids  # no chars missing from the corpus-built vocab


class TestVLMNet:
    def _model(self):
        tok = build_tokenizer()
        torch.manual_seed(0)
        return tok, VLMNet(vocab_size=len(tok.stoi), d_model=32, n_heads=2,
                           n_layers=2, max_text_len=40)

    def test_forward_shape(self):
        tok, model = self._model()
        logits = model(torch.randn(2, 1, 28, 28), torch.zeros(2, 40, dtype=torch.long))
        assert logits.shape == (2, 40, len(tok.stoi))

    def test_image_changes_output(self):
        """The prefix path is wired: different images, different text logits."""
        tok, model = self._model()
        model.eval()
        tokens = torch.zeros(1, 10, dtype=torch.long)
        with torch.no_grad():
            l1 = model(torch.zeros(1, 1, 28, 28), tokens)
            l2 = model(torch.ones(1, 1, 28, 28), tokens)
        assert not torch.allclose(l1, l2, atol=1e-5)

    def test_answer_terminates(self):
        tok, model = self._model()
        outs = model.answer(torch.randn(3, 1, 28, 28), tok,
                            QA_TEMPLATES[0][0], max_new=8)
        assert len(outs) == 3
        assert all(isinstance(o, str) and "." not in o for o in outs)


class TestVLMSmoke:
    def test_train_all_s_tier(self):
        config = VLMConfig(fast_demo=True, data_root=DATA_ROOT, epochs=1,
                           eval_samples=8, d_model=32, n_layers=2, n_heads=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_vlm")
            result = VLM().train_all(config, logger)
            assert 0.0 <= result["answer_accuracy"] <= 1.0
            assert 0.0 <= result["blind_accuracy"] <= 1.0
            assert logger.artifact_path("model.pt").exists()
            keys = {m.get("key") for m in logger.read_metrics()}
            assert "loss" in keys
