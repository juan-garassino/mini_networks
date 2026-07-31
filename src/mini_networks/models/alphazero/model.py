"""Mini AlphaZero: a planner that teaches its own evaluator.

Key idea (Silver et al., arXiv 1712.01815): close the loop between SEARCH
and LEARNING. MCTS guided by a policy/value network produces move
distributions that are STRONGER than the raw network (search is a policy
improvement operator); train the network to imitate those distributions and
predict the self-play outcome, and the next round of search — leaning on a
better network — is stronger still. No human data, no handcrafted features:
the game rules plus this loop.

This implementation (defaults): tic-tac-toe (the minimal domain where the
loop is honest), an MLP policy+value net on an 18-dim two-plane encoding
(always from the player-to-move's perspective), PUCT-guided MCTS with
Dirichlet root noise during self-play. The QUALITY GATE evaluates the RAW
policy without search — MCTS with terminal backups beats a random opponent
even with an untrained net, so gating search-assisted play would measure
the planner, not the learning.

Key equations:
  PUCT     a* = argmax_a Q(s,a) + c_puct P(s,a) sqrt(sum_b N(s,b)) / (1 + N(s,a))
  targets  policy <- MCTS visit distribution;  value <- game outcome,
           SIGN-FLIPPED to each stored position's player-to-move perspective
  loss     CE(policy, visits) + MSE(value, z)

Deliberately simplified vs the paper: an MLP instead of a ResNet, no
resignation, no evaluation-gated checkpointing, tic-tac-toe instead of Go —
the closed improvement loop is the payload, and it forces the classic bugs
(value perspective, legality masking, noise-at-eval) into the open.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_networks.models.alphazero.game import (
    apply_move,
    encode,
    legal_moves,
    winner,
)


class PolicyValueNet(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(18, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, 9)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy_head(h), torch.tanh(self.value_head(h)).squeeze(-1)

    @torch.no_grad()
    def priors_value(self, board: tuple, player: int) -> tuple[torch.Tensor, float]:
        """Masked, renormalized move priors + value for the player to move."""
        logits, value = self(encode(board, player).unsqueeze(0))
        mask = torch.full((9,), float("-inf"))
        legal = legal_moves(board)
        mask[legal] = 0.0
        priors = F.softmax(logits[0] + mask, dim=-1)
        return priors, float(value.item())


class _Node:
    __slots__ = ("prior", "visits", "value_sum", "children")

    def __init__(self, prior: float):
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children: dict[int, _Node] = {}

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class MCTS:
    """PUCT search. Values are always from the perspective of the player to
    move at each node; backups negate per ply."""

    def __init__(self, net: PolicyValueNet, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.6, dirichlet_eps: float = 0.25):
        self.net = net
        self.c_puct = c_puct
        self.alpha = dirichlet_alpha
        self.eps = dirichlet_eps

    def run(self, board: tuple, player: int, n_sims: int,
            add_noise: bool = False, g: torch.Generator | None = None) -> torch.Tensor:
        """Returns the root visit distribution over the 9 moves."""
        root = _Node(prior=1.0)
        self._expand(root, board, player)
        if add_noise and root.children:
            noise = torch.distributions.Dirichlet(
                torch.full((len(root.children),), self.alpha)
            ).sample()
            for n, (move, child) in zip(noise, root.children.items()):
                child.prior = (1 - self.eps) * child.prior + self.eps * float(n)

        for _ in range(n_sims):
            node, b, p = root, board, player
            path = [node]
            # select down to a leaf
            while node.children:
                move, node = self._select(node)
                b = apply_move(b, move, p)
                p = -p
                path.append(node)
            w = winner(b)
            if w is None:
                value = self._expand(node, b, p)
            else:
                # terminal: outcome from the perspective of the player to move at b
                value = 0.0 if w == 0 else (1.0 if w == p else -1.0)
            # backup, flipping sign each ply (parent's mover is the other player)
            for n in reversed(path):
                n.visits += 1
                n.value_sum += value
                value = -value

        visits = torch.zeros(9)
        for move, child in root.children.items():
            visits[move] = child.visits
        return visits / visits.sum() if visits.sum() > 0 else visits

    def _select(self, node: _Node) -> tuple[int, _Node]:
        total = math.sqrt(sum(c.visits for c in node.children.values()) + 1e-8)
        best, best_score = None, -float("inf")
        for move, child in node.children.items():
            # child.q is from the CHILD mover's perspective — negate for the parent
            score = -child.q + self.c_puct * child.prior * total / (1 + child.visits)
            if score > best_score:
                best, best_score = move, score
        return best, node.children[best]

    def _expand(self, node: _Node, board: tuple, player: int) -> float:
        priors, value = self.net.priors_value(board, player)
        for move in legal_moves(board):
            node.children[move] = _Node(prior=float(priors[move]))
        return value
