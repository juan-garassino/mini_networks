from __future__ import annotations
from mini_networks.core.config import BaseConfig


class ReinforceConfig(BaseConfig):
    model_name: str = "reinforce"

    # Maze
    maze_width: int = 8
    maze_height: int = 8
    maze_density: float = 0.2
    max_steps: int = 200

    # Policy network
    hidden_dim: int = 64

    # REINFORCE
    gamma: float = 0.99

    # Training
    n_episodes: int = 500
    eval_episodes: int = 20
