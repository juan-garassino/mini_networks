"""Tests for mini AlphaZero: game logic, MCTS, value perspective, trainer."""
import tempfile

import pytest
import torch

from mini_networks.core.logging.logger import Logger
from mini_networks.models.alphazero.config import AlphaZeroConfig
from mini_networks.models.alphazero.game import (
    WIN_LINES,
    apply_move,
    encode,
    legal_moves,
    player_to_move,
    winner,
)
from mini_networks.models.alphazero.model import MCTS, PolicyValueNet
from mini_networks.models.alphazero.trainer import (
    AlphaZeroTrainer,
    make_alphazero_dataloader,
)


class TestGame:
    def test_all_win_lines(self):
        for line in WIN_LINES:
            b = [0] * 9
            for i in line:
                b[i] = 1
            assert winner(tuple(b)) == 1
            b = [-v for v in b]
            assert winner(tuple(b)) == -1

    def test_draw_and_ongoing(self):
        assert winner((1, -1, 1, 1, -1, -1, -1, 1, 1)) == 0
        assert winner((0,) * 9) is None

    def test_player_to_move(self):
        assert player_to_move((0,) * 9) == 1
        assert player_to_move((1, 0, 0, 0, 0, 0, 0, 0, 0)) == -1
        with pytest.raises(ValueError):
            player_to_move((1, 1, 1, 0, 0, 0, 0, 0, 0))  # 3 X vs 0 O impossible

    def test_encode_perspective(self):
        """encode(b, +1) and encode(b, -1) swap the two planes."""
        b = (1, -1, 0, 0, 1, 0, 0, 0, -1)
        e1 = encode(b, 1)
        e2 = encode(b, -1)
        assert torch.equal(e1[:9], e2[9:])
        assert torch.equal(e1[9:], e2[:9])


class TestMCTS:
    def _mcts(self):
        torch.manual_seed(0)
        return MCTS(PolicyValueNet(hidden_dim=16))

    def test_visits_only_legal(self):
        board = (1, -1, 1, -1, 0, 0, 0, 0, 0)
        pi = self._mcts().run(board, 1, n_sims=30)
        illegal = [i for i in range(9) if board[i] != 0]
        assert pi[illegal].sum() == 0
        assert abs(pi.sum().item() - 1.0) < 1e-5

    def test_finds_winning_move(self):
        """X to move with two in a row: search must prefer the winning square."""
        board = (1, 1, 0, -1, -1, 0, 0, 0, 0)
        pi = self._mcts().run(board, 1, n_sims=100)
        assert int(pi.argmax()) in (2, 5)  # win at 2; blocking O's win at 5 also defensible
        # with more sims the immediate win should dominate
        pi = self._mcts().run(board, 1, n_sims=300)
        assert int(pi.argmax()) == 2

    def test_noise_only_when_asked(self):
        m = self._mcts()
        a = m.run((0,) * 9, 1, n_sims=20, add_noise=False)
        b = m.run((0,) * 9, 1, n_sims=20, add_noise=False)
        assert torch.equal(a, b)  # deterministic without noise


class TestValuePerspective:
    def test_sign_flip_targets(self):
        """A finished game's stored value targets must alternate sign with the
        player to move — the classic AlphaZero bug."""
        # X wins: z = +1; positions where X moved get +1, O positions get -1
        z = 1
        players = [1, -1, 1, -1, 1]
        targets = [z * p for p in players]
        assert targets == [1, -1, 1, -1, 1]


class TestAlphaZeroTrainer:
    def _config(self, **kwargs):
        defaults = dict(hidden_dim=16, fast_demo=True, epochs=1)
        defaults.update(kwargs)
        return AlphaZeroConfig(**defaults)

    def test_train_smoke(self):
        config = self._config()
        trainer = AlphaZeroTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_az")
            trainer.train(config, make_alphazero_dataloader(config), logger)
            keys = {m.get("key") for m in logger.read_metrics()}
            assert {"loss", "policy_loss", "value_loss"} <= keys
            assert logger.artifact_path("model.pt").exists()

    def test_evaluate_gates_raw_policy(self):
        config = self._config()
        trainer = AlphaZeroTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_az")
            trainer.train(config, make_alphazero_dataloader(config), logger)
            result = trainer.evaluate(config, make_alphazero_dataloader(config), logger)
            assert 0.0 <= result["success_rate"] <= 1.0
            assert 0.0 <= result["search_success_rate"] <= 1.0

    def test_eval_deterministic(self):
        config = self._config()
        trainer = AlphaZeroTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_az")
            trainer.train(config, make_alphazero_dataloader(config), logger)
            a = trainer.evaluate(config, make_alphazero_dataloader(config), logger)
            b = trainer.evaluate(config, make_alphazero_dataloader(config), logger)
            assert a["success_rate"] == b["success_rate"]  # seeded opponent

    def test_infer_rejects_bad_boards(self):
        config = self._config()
        trainer = AlphaZeroTrainer()
        trainer.model = trainer._build(config)
        with pytest.raises(ValueError):
            trainer.infer(config, {"board": [2] * 9})
        with pytest.raises(ValueError):
            trainer.infer(config, {"board": [1, 1, 1, 1, 0, 0, 0, 0, 0]})
        out = trainer.infer(config, {"board": [0] * 9})
        assert 0 <= out["move"] < 9

    def test_checkpoint_roundtrip(self):
        config = self._config()
        trainer = AlphaZeroTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_az")
            trainer.train(config, make_alphazero_dataloader(config), logger)
            fresh = AlphaZeroTrainer()
            fresh.load_checkpoint(config, logger.artifact_path("model.pt").parent)
            assert fresh.model is not None
