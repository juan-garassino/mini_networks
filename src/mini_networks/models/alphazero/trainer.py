"""AlphaZero trainer: self-play episodes, raw-policy gate, search evidence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mini_networks.core.config import BaseConfig
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.alphazero.config import AlphaZeroConfig
from mini_networks.models.alphazero.game import (
    apply_move,
    encode,
    legal_moves,
    player_to_move,
    winner,
)
from mini_networks.models.alphazero.model import MCTS, PolicyValueNet

import logging

log = logging.getLogger(__name__)


class AlphaZeroTrainer(BaseTrainer):
    def __init__(self):
        self.model: PolicyValueNet | None = None

    def _build(self, config: AlphaZeroConfig) -> PolicyValueNet:
        # CPU on purpose: MCTS issues thousands of single-position forwards,
        # which are GPU-launch-latency-bound — CPU is faster here.
        return PolicyValueNet(hidden_dim=config.hidden_dim)

    def _raw_move(self, board: tuple, player: int) -> int:
        """Gate-path move: raw policy head, legality-masked, NO search."""
        with torch.no_grad():
            logits, _ = self.model(encode(board, player).unsqueeze(0))
        mask = torch.full((9,), float("-inf"))
        mask[legal_moves(board)] = 0.0
        return int((logits[0] + mask).argmax())

    def _play_vs_random(self, games: int, g: torch.Generator,
                        use_search: bool = False, config=None) -> float:
        """(wins + 0.5 draws) / games vs seeded random; sides alternate 50/50."""
        mcts = MCTS(self.model, config.c_puct) if use_search else None
        score = 0.0
        for gi in range(games):
            we_are = 1 if gi % 2 == 0 else -1
            board, player = (0,) * 9, 1
            while winner(board) is None:
                if player == we_are:
                    if use_search:
                        sims = config.limit_steps(config.n_sims, s_cap=8, m_cap=64)
                        visits = mcts.run(board, player, sims, add_noise=False)
                        move = int(visits.argmax())
                    else:
                        move = self._raw_move(board, player)
                else:
                    legal = legal_moves(board)
                    move = legal[int(torch.randint(0, len(legal), (1,), generator=g))]
                board = apply_move(board, move, player)
                player = -player
            w = winner(board)
            score += 1.0 if w == we_are else (0.5 if w == 0 else 0.0)
        return score / max(1, games)

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, AlphaZeroConfig)
        del dataloader  # self-play generates its own experience (rl_maze idiom)
        model = self._build(config)
        self.model = model
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        mcts = MCTS(model, config.c_puct, config.dirichlet_alpha, config.dirichlet_eps)
        logger.log_config(config.model_dump())
        g = torch.Generator().manual_seed(config.seed)

        n_games = config.limit_steps(config.n_games, s_cap=2, m_cap=200)
        n_sims = config.limit_steps(config.n_sims, s_cap=8, m_cap=64)
        buffer: list[tuple[torch.Tensor, torch.Tensor, int]] = []  # (state, pi, player)
        outcomes: list[float] = []

        for game_i in range(n_games):
            board, player = (0,) * 9, 1
            records = []
            while winner(board) is None:
                pi = mcts.run(board, player, n_sims, add_noise=True)
                records.append((encode(board, player), pi, player))
                move = int(torch.multinomial(pi + 1e-9, 1, generator=g))
                board = apply_move(board, move, player)
                player = -player
            z = winner(board)  # from X's (+1) perspective
            for state, pi, p in records:
                # value target SIGN-FLIPPED to the stored player-to-move's view
                buffer.append((state, pi, float(z * p)))
            buffer = buffer[-config.buffer_size:]
            outcomes.append(float(z))

            p_loss_avg = v_loss_avg = 0.0
            for _ in range(config.train_steps_per_game):
                # replacement: at S the buffer (~10-18 positions) is smaller than a batch
                idx = torch.randint(0, len(buffer), (min(64, len(buffer) * 2),), generator=g)
                states = torch.stack([buffer[i][0] for i in idx])
                pis = torch.stack([buffer[i][1] for i in idx])
                zs = torch.tensor([buffer[i][2] for i in idx])
                logits, values = model(states)
                policy_loss = -(pis * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                value_loss = F.mse_loss(values, zs)
                loss = policy_loss + value_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                p_loss_avg += policy_loss.item() / config.train_steps_per_game
                v_loss_avg += value_loss.item() / config.train_steps_per_game

            logger.log_metrics(game_i, {
                "loss": p_loss_avg + v_loss_avg,
                "policy_loss": p_loss_avg,
                "value_loss": v_loss_avg,
            })
            if game_i % 25 == 0:
                log.info(f"  game {game_i}  loss {p_loss_avg + v_loss_avg:.4f}")

        torch.save(model.state_dict(), logger.artifact_path("model.pt"))

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        """Gate = RAW policy vs seeded random (search would win even untrained);
        search_success_rate is logged as the search-alone reference."""
        assert isinstance(config, AlphaZeroConfig)
        if self.model is None:
            self.model = self._build(config)
        games = config.limit_steps(config.eval_games, s_cap=8, m_cap=200)
        g = torch.Generator().manual_seed(1234)
        raw = self._play_vs_random(games, g, use_search=False, config=config)
        g = torch.Generator().manual_seed(1234)
        search = self._play_vs_random(max(8, games // 5), g, use_search=True, config=config)
        return {"success_rate": raw, "search_success_rate": search}

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, AlphaZeroConfig)
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        board = tuple(inputs.get("board", [0] * 9)) if isinstance(inputs, dict) else (0,) * 9
        if len(board) != 9 or any(v not in (-1, 0, 1) for v in board):
            raise ValueError("board must be 9 ints in {-1, 0, 1}")
        player = player_to_move(board)  # raises on impossible piece counts
        with torch.no_grad():
            logits, value = self.model(encode(board, player).unsqueeze(0))
        mask = torch.full((9,), float("-inf"))
        mask[legal_moves(board)] = 0.0
        policy = F.softmax(logits[0] + mask, dim=-1)
        return {
            "move": int(policy.argmax()),
            "policy": policy.tolist(),
            "value": float(value.item()),
            "player": player,
        }

    def load_checkpoint(self, config: BaseConfig, artifacts_dir) -> None:
        assert isinstance(config, AlphaZeroConfig)
        state = torch.load(Path(artifacts_dir) / "model.pt", map_location="cpu",
                           weights_only=True)
        self.model = self._build(config)
        self.model.load_state_dict(state)
        self.model.eval()


def make_alphazero_dataloader(config: AlphaZeroConfig, split: str = "train") -> DataLoader:
    # self-play generates its own data; dummy loader keeps the registry contract
    return DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)
