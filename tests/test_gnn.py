"""Tests for the GCN: SBM dataset hygiene, message passing, transductive trainer."""
import tempfile

import torch

from mini_networks.core.data.registry import SyntheticSBMGraph
from mini_networks.core.logging.logger import Logger
from mini_networks.models.gnn.config import GNNConfig
from mini_networks.models.gnn.model import GCN
from mini_networks.models.gnn.trainer import GNNTrainer, make_gnn_dataloader


class TestSyntheticSBMGraph:
    def test_deterministic(self):
        a = SyntheticSBMGraph(seed=11)
        b = SyntheticSBMGraph(seed=11)
        assert torch.equal(a._x, b._x)
        assert torch.equal(a._a_norm, b._a_norm)
        assert torch.equal(a._train_mask, b._train_mask)

    def test_stratified_masks(self):
        ds = SyntheticSBMGraph(n_nodes=200, n_communities=4, train_per_class=5)
        x, a, y, train_mask, test_mask = ds[0]
        assert train_mask.sum() == 20
        for c in range(4):
            assert (train_mask & (y == c)).sum() == 5  # exactly 5 per community
        assert not (train_mask & test_mask).any()
        assert (train_mask | test_mask).all()

    def test_a_norm_self_loops_finite(self):
        ds = SyntheticSBMGraph(n_nodes=100)
        _, a, _, _, _ = ds[0]
        assert torch.isfinite(a).all()  # self-loops kill isolated-node 0/0
        assert (a.diagonal() > 0).all()
        assert torch.allclose(a, a.T, atol=1e-6)  # symmetric normalization

    def test_community_structure_in_adjacency(self):
        """Intra-community edge density must dominate inter — the whole premise."""
        ds = SyntheticSBMGraph(n_nodes=200, p_in=0.15, p_out=0.02)
        y = ds._y
        same = y.unsqueeze(0) == y.unsqueeze(1)
        adj = ds.adjacency
        intra = adj[same].mean()
        inter = adj[~same].mean()
        assert intra > 3 * inter

    def test_single_item(self):
        ds = SyntheticSBMGraph()
        assert len(ds) == 1


class TestGCN:
    def test_forward_shape(self):
        model = GCN(n_features=8, hidden_dim=16, n_classes=4)
        x = torch.randn(50, 8)
        a = torch.eye(50)
        assert model(x, a).shape == (50, 4)

    def test_propagation_matters(self):
        """With identity adjacency vs real adjacency the outputs must differ —
        the graph is actually used."""
        torch.manual_seed(0)
        model = GCN(n_features=8, hidden_dim=16, n_classes=4, dropout=0.0)
        model.eval()
        ds = SyntheticSBMGraph(n_nodes=50)
        x, a, _, _, _ = ds[0]
        with torch.no_grad():
            out_graph = model(x, a)
            out_iso = model(x, torch.eye(50))
        assert not torch.allclose(out_graph, out_iso, atol=1e-4)


class TestGNNTrainer:
    def _config(self, **kwargs):
        defaults = dict(n_nodes=100, fast_demo=True, epochs=1, hidden_dim=16)
        defaults.update(kwargs)
        return GNNConfig(**defaults)

    def test_loader_single_batch_squeeze(self):
        config = self._config()
        dl = make_gnn_dataloader(config)
        batch = next(iter(dl))
        assert batch[0].shape == (1, 100, 8)  # leading batch dim, squeezed in trainer

    def test_train_smoke(self):
        config = self._config()
        trainer = GNNTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_gnn")
            trainer.train(config, make_gnn_dataloader(config), logger)
            assert len(logger.read_metrics()) > 0
            assert logger.artifact_path("model.pt").exists()

    def test_evaluate_reports_baseline(self):
        config = self._config()
        trainer = GNNTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_gnn")
            trainer.train(config, make_gnn_dataloader(config), logger)
            result = trainer.evaluate(config, make_gnn_dataloader(config), logger)
            assert 0.0 <= result["accuracy"] <= 1.0
            assert 0.0 <= result["mlp_baseline_accuracy"] <= 1.0

    def test_infer(self):
        config = self._config()
        trainer = GNNTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_gnn")
            trainer.train(config, make_gnn_dataloader(config), logger)
            out = trainer.infer(config, {"node_id": 3})
            assert set(out) == {"node_id", "predicted", "true", "n_neighbors"}
            assert 0 <= out["predicted"] < 4

    def test_checkpoint_roundtrip(self):
        config = self._config()
        trainer = GNNTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_gnn")
            trainer.train(config, make_gnn_dataloader(config), logger)
            fresh = GNNTrainer()
            fresh.load_checkpoint(config, logger.artifact_path("model.pt").parent)
            assert fresh.model is not None
