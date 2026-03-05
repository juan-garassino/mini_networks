"""Smoke tests for NanoRAG: TF-IDF retrieval + TransformerLM generation."""
import os
import tempfile

import torch
import pytest

from mini_networks.models.rag.config import RAGConfig
from mini_networks.models.rag.model import NanoRAG, TFIDFIndex
from mini_networks.models.rag.trainer import RAGTrainer, make_rag_dataloader
from mini_networks.models.transformer.model import TransformerLM
from mini_networks.models.transformer.tokenizer import CharTokenizer
from mini_networks.core.logging.logger import Logger

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")

DOCS = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast red cat leaps across the sleeping hound.",
    "Machine learning models learn from data.",
    "Neural networks are used in deep learning research.",
    "Shakespeare wrote many famous plays and sonnets.",
]


# ---------------------------------------------------------------------------
# TFIDFIndex
# ---------------------------------------------------------------------------

class TestTFIDFIndex:
    def test_build_no_error(self):
        index = TFIDFIndex()
        index.build(DOCS)
        assert index._tfidf is not None
        assert index._tfidf.shape[0] == len(DOCS)

    def test_retrieve_returns_k_results(self):
        index = TFIDFIndex()
        index.build(DOCS)
        results = index.retrieve("machine learning", top_k=2)
        assert len(results) == 2

    def test_retrieve_scores_are_float(self):
        index = TFIDFIndex()
        index.build(DOCS)
        results = index.retrieve("dog fox", top_k=3)
        for score, doc in results:
            assert isinstance(score, float)
            assert isinstance(doc, str)

    def test_retrieve_most_relevant_first(self):
        index = TFIDFIndex()
        index.build(DOCS)
        results = index.retrieve("machine learning neural networks", top_k=2)
        scores = [s for s, _ in results]
        assert scores[0] >= scores[1]

    def test_retrieve_empty_query_returns_empty(self):
        index = TFIDFIndex()
        index.build(DOCS)
        results = index.retrieve("", top_k=3)
        assert results == []

    def test_retrieve_on_empty_index_returns_empty(self):
        index = TFIDFIndex()
        results = index.retrieve("anything", top_k=2)
        assert results == []

    def test_idf_shape(self):
        index = TFIDFIndex()
        index.build(DOCS)
        V = len(index._word2idx)
        assert index._idf.shape == (V,)

    def test_single_document(self):
        index = TFIDFIndex()
        index.build(["hello world test"])
        results = index.retrieve("hello", top_k=1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# NanoRAG
# ---------------------------------------------------------------------------

class TestNanoRAG:
    def _rag(self):
        rag = NanoRAG(top_k=2, chunk_size=100)
        rag.add_documents(DOCS)
        return rag

    def test_add_documents_creates_index(self):
        rag = self._rag()
        assert rag.index._tfidf is not None

    def test_retrieve_returns_strings(self):
        rag = self._rag()
        results = rag.retrieve("fox dog")
        assert isinstance(results, list)
        assert all(isinstance(r, str) for r in results)

    def test_retrieve_top_k_respected(self):
        rag = self._rag()
        results = rag.retrieve("fox", top_k=3)
        assert len(results) <= 3

    def test_build_prompt_contains_context(self):
        rag = self._rag()
        prompt = rag.build_prompt("fox dog")
        assert "[Context:" in prompt
        assert "fox dog" in prompt

    def test_build_prompt_empty_docs_returns_query(self):
        rag = NanoRAG(top_k=2)
        prompt = rag.build_prompt("hello")
        assert prompt == "hello"

    def test_generate_returns_string(self):
        rag = self._rag()
        # Build a tiny tokenizer + model
        text = " ".join(DOCS)
        tokenizer = CharTokenizer.from_text(text)
        model = TransformerLM(
            vocab_size=tokenizer.vocab_size,
            d_model=32, n_heads=2, n_layers=1, d_ff=64, seq_len=64,
        )
        result = rag.generate("fox", model, tokenizer, max_new_tokens=8)
        assert isinstance(result, str)

    def test_generate_length_bounded(self):
        rag = self._rag()
        text = " ".join(DOCS)
        tokenizer = CharTokenizer.from_text(text)
        model = TransformerLM(
            vocab_size=tokenizer.vocab_size,
            d_model=32, n_heads=2, n_layers=1, d_ff=64, seq_len=64,
        )
        result = rag.generate("fox", model, tokenizer, max_new_tokens=5)
        # Should produce some output
        assert len(result) > 0


# ---------------------------------------------------------------------------
# RAGTrainer
# ---------------------------------------------------------------------------

class TestRAGTrainer:
    def _config(self, **kwargs):
        defaults = dict(
            d_model=32, n_layers=1, n_heads=2, d_ff=64,
            seq_len=16, fast_demo=True, data_root=DATA_ROOT,
            epochs=1, top_k=2, chunk_size=100,
        )
        defaults.update(kwargs)
        return RAGConfig(**defaults)

    def test_train_smoke(self):
        config = self._config()
        trainer = RAGTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_rag")
            dl = make_rag_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            assert len(metrics) > 0

    def test_checkpoint_saved(self):
        config = self._config()
        trainer = RAGTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_rag_ckpt")
            dl = make_rag_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            assert logger.artifact_path("model.pt").exists()

    def test_evaluate(self):
        config = self._config()
        trainer = RAGTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_rag_eval")
            dl = make_rag_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.evaluate(config, dl, logger)
            assert "eval_loss" in result

    def test_infer_returns_generated_and_retrieved(self):
        config = self._config()
        trainer = RAGTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_rag_infer")
            dl = make_rag_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            result = trainer.infer(config, {"query": "KING", "max_new_tokens": 8})
            assert "generated" in result
            assert "retrieved" in result
            assert isinstance(result["generated"], str)
            assert isinstance(result["retrieved"], list)

    def test_rag_index_built_after_train(self):
        config = self._config()
        trainer = RAGTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_rag_index")
            dl = make_rag_dataloader(config, split="train")
            trainer.train(config, dl, logger)
            assert trainer.rag is not None
            assert trainer.rag.index._tfidf is not None
