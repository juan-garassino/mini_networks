"""Procedurally-generated grid maze environment.

Grid codes
----------
  0 = HOLE (obstacle / trap)
  1 = PATH (walkable)
  2 = START
  3 = GOAL (treasure)

State representation
--------------------
  Flattened 5×5 neighbourhood around the agent (clamped at boundaries)
  concatenated with the agent's (x, y) position → 27-dim float vector.

Action space
------------
  0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT

Rewards
-------
  +1.0  reaching the goal
  -0.5  stepping into a hole (also terminates episode)
  -0.01 each step (encourages efficiency)

Educational notes
-----------------
  - No gym dependency: clean stand-alone implementation.
  - Maze is regenerated on every reset() when `random_reset=True`.
  - `render()` prints an ASCII view for quick inspection.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

HOLE  = 0
PATH  = 1
START = 2
GOAL  = 3

ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # dy, dx


class MazeEnv:
    """Simple grid maze with procedural generation."""

    N_ACTIONS = 4

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        density: float = 0.2,
        max_steps: int = 200,
        seed: Optional[int] = None,
        random_reset: bool = False,
    ):
        self.width = width
        self.height = height
        self.density = density
        self.max_steps = max_steps
        self.random_reset = random_reset
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        self.maze: np.ndarray = np.array([])
        self.start_pos: tuple[int, int] = (0, 0)
        self.goal_pos: tuple[int, int] = (0, 0)
        self.agent_pos: tuple[int, int] = (0, 0)
        self._steps = 0

        self._generate()

    # ------------------------------------------------------------------
    # Maze generation
    # ------------------------------------------------------------------

    def _generate(self) -> None:
        """Generate a new maze ensuring start and goal are PATH cells."""
        maze = np.ones((self.height, self.width), dtype=np.int32)

        # Scatter random holes (avoid corners for start/goal)
        for r in range(self.height):
            for c in range(self.width):
                if self._np_rng.random() < self.density:
                    maze[r, c] = HOLE

        # Place start (top-left area) and goal (bottom-right area)
        sy, sx = 0, 0
        gy, gx = self.height - 1, self.width - 1
        maze[sy, sx] = START
        maze[gy, gx] = GOAL

        # Ensure a minimal path exists (simple: clear the diagonal)
        steps = max(self.height, self.width)
        for i in range(steps):
            r = min(int(i * self.height / steps), self.height - 1)
            c = min(int(i * self.width / steps), self.width - 1)
            if maze[r, c] == HOLE:
                maze[r, c] = PATH

        self.maze = maze
        self.start_pos = (sy, sx)
        self.goal_pos = (gy, gx)

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------

    @property
    def state_size(self) -> int:
        return 25 + 2  # 5×5 neighbourhood + (row, col)

    def reset(self) -> np.ndarray:
        if self.random_reset:
            self._generate()
        self.agent_pos = self.start_pos
        self._steps = 0
        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        dy, dx = ACTIONS[action]
        r, c = self.agent_pos
        nr, nc = r + dy, c + dx

        # Clamp to grid boundary — hitting a wall costs a step but stays
        nr = max(0, min(self.height - 1, nr))
        nc = max(0, min(self.width  - 1, nc))

        self.agent_pos = (nr, nc)
        self._steps += 1

        cell = self.maze[nr, nc]
        if cell == GOAL:
            reward, done = 1.0, True
        elif cell == HOLE:
            reward, done = -0.5, True
        else:
            reward, done = -0.01, False

        if self._steps >= self.max_steps:
            done = True

        info = {"cell": int(cell), "steps": self._steps}
        return self._get_state(), reward, done, info

    # ------------------------------------------------------------------
    # State representation
    # ------------------------------------------------------------------

    def _get_state(self) -> np.ndarray:
        r, c = self.agent_pos
        view = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.height and 0 <= cc < self.width:
                    view.append(float(self.maze[rr, cc]))
                else:
                    view.append(float(HOLE))  # out-of-bounds = obstacle
        return np.array(view + [float(r), float(c)], dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Return ASCII representation of the current maze state."""
        symbols = {HOLE: "X", PATH: ".", START: "S", GOAL: "G"}
        rows = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                if (r, c) == self.agent_pos:
                    row.append("@")
                else:
                    row.append(symbols.get(self.maze[r, c], "?"))
            rows.append(" ".join(row))
        return "\n".join(rows)
