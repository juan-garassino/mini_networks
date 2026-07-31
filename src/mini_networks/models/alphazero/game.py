"""Tic-tac-toe: the minimal honest domain for search-guided self-play.

Board: tuple of 9 ints, 0 empty / +1 X / -1 O; X always moves first.
All state encodings are from the PERSPECTIVE OF THE PLAYER TO MOVE — plane 0
is "my pieces", plane 1 is "opponent pieces" — so one network plays both
sides and values are always "how good for whoever moves now".
"""
from __future__ import annotations

import torch

WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
)


def legal_moves(board: tuple) -> list[int]:
    return [i for i, v in enumerate(board) if v == 0]


def apply_move(board: tuple, move: int, player: int) -> tuple:
    assert board[move] == 0
    b = list(board)
    b[move] = player
    return tuple(b)


def winner(board: tuple) -> int | None:
    """+1 / -1 if that player won, 0 for a draw, None if the game continues."""
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    if all(v != 0 for v in board):
        return 0
    return None


def player_to_move(board: tuple) -> int:
    """X (+1) moves when piece counts are equal; O (-1) otherwise."""
    x = sum(1 for v in board if v == 1)
    o = sum(1 for v in board if v == -1)
    if x not in (o, o + 1):
        raise ValueError(f"impossible board: {x} X vs {o} O")
    return 1 if x == o else -1


def encode(board: tuple, player: int) -> torch.Tensor:
    """[18] float: my pieces then opponent pieces, from `player`'s perspective."""
    b = torch.tensor(board, dtype=torch.float32)
    return torch.cat([(b == player).float(), (b == -player).float()])
