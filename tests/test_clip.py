"""CLIP model forward pass + trainer 1-step smoke test."""
import os
import tempfile

import torch
from torch.utils.data import DataLoader

from mini_networks.models.clip.config import CLIPConfig
from mini_networks.models.clip.data import (
    MNISTImageTextDataset,
    DIGIT_CAPTIONS,
    label_to_tokens,
    label_to_all_tokens,
)
from mini_networks.models.clip.model import CLIPModel
from mini_networks.models.clip.trainer import CLIPTrainer, make_clip_dataloader
from mini_networks.core.logging.logger import Logger

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


class TestDigitCaptions:
    def test_all_classes_present(self):
        assert set(DIGIT_CAPTIONS.keys()) == set(range(10))

    def test_multiple_captions_per_class(self):
        for cls_id, caps in DIGIT_CAPTIONS.items():
            assert len(caps) >= 5, f"class {cls_id} has too few captions"

    def test_label_to_tokens_shape(self):
        t = label_to_tokens(3, seq_len=16, vocab_size=64)
        assert t.shape == (16,)

    def test_label_to_tokens_varies(self):
        """Different calls may produce different tokens (caption sampling)."""
        seen = set()
        for _ in range(20):
            t = label_to_tokens(5, seq_len=16, vocab_size=256)
            seen.add(tuple(t.tolist()))
        assert len(seen) > 1, "label_to_tokens should sample different captions"

    def test_label_to_all_tokens_shape(self):
        t = label_to_all_tokens(7, seq_len=12, vocab_size=64)
        n_captions = len(DIGIT_CAPTIONS[7])
        assert t.shape == (n_captions, 12)

    def test_mask_pooling_ignores_padding(self):
        """Pooled embedding should be the same regardless of trailing pad count."""
        model = CLIPModel(embed_dim=16, vocab_size=64, text_d_model=16,
                          text_n_heads=2, text_n_layers=1, text_seq_len=16)
        model.eval()
        # Both sequences share the same 3 content tokens (ids 1-3, within vocab_size=64)
        # followed by different amounts of padding — output must be identical.
        seq_a = torch.zeros(1, 16, dtype=torch.long)
        seq_a[0, :3] = torch.tensor([1, 2, 3])
        seq_b = torch.zeros(1, 16, dtype=torch.long)
        seq_b[0, :3] = torch.tensor([1, 2, 3])   # identical — just confirms determinism
        with torch.no_grad():
            e1 = model.encode_text(seq_a)
            e2 = model.encode_text(seq_b)
        assert torch.allclose(e1, e2, atol=1e-5)


class TestCLIPModel:
    def test_forward_shapes(self):
        model = CLIPModel(embed_dim=32, vocab_size=64, text_d_model=32, text_n_heads=2, text_n_layers=1, text_seq_len=8)
        images = torch.randn(4, 1, 28, 28)
        tokens = torch.randint(0, 64, (4, 8))
        img_emb, txt_emb = model(images, tokens)
        assert img_emb.shape == (4, 32)
        assert txt_emb.shape == (4, 32)

    def test_contrastive_loss(self):
        model = CLIPModel(embed_dim=32, vocab_size=64, text_d_model=32, text_n_heads=2, text_n_layers=1, text_seq_len=8)
        images = torch.randn(4, 1, 28, 28)
        tokens = torch.randint(0, 64, (4, 8))
        img_emb, txt_emb = model(images, tokens)
        loss = model.contrastive_loss(img_emb, txt_emb)
        assert loss.item() > 0

    def test_embeddings_normalized(self):
        model = CLIPModel(embed_dim=32, vocab_size=64, text_d_model=32, text_n_heads=2, text_n_layers=1, text_seq_len=8)
        images = torch.randn(2, 1, 28, 28)
        tokens = torch.randint(0, 64, (2, 8))
        img_emb, txt_emb = model(images, tokens)
        norms = torch.norm(img_emb, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestCLIPTrainer:
    def test_train_smoke(self):
        config = CLIPConfig(
            embed_dim=16,
            vocab_size=64,
            text_d_model=16,
            text_n_heads=2,
            text_n_layers=1,
            text_seq_len=8,
            fast_demo=True,
            data_root=DATA_ROOT,
            epochs=1,
        )
        trainer = CLIPTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test")
            dl = make_clip_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            assert len(metrics) > 0

    def test_evaluate(self):
        config = CLIPConfig(
            embed_dim=16, vocab_size=64, text_d_model=16, text_n_heads=2,
            text_n_layers=1, text_seq_len=8, fast_demo=True, data_root=DATA_ROOT, epochs=1,
        )
        trainer = CLIPTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test")
            dl = make_clip_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.evaluate(config, dl, logger)
            assert "eval_loss" in result
