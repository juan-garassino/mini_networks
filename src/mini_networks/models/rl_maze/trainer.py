"""RLMazeTrainer — trains Q, DQN, or PPO agents on MazeEnv."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.rl_maze.config import RLMazeConfig
from mini_networks.models.rl_maze.env import MazeEnv
from mini_networks.models.rl_maze.agents import QAgent, DQNAgent, PPOAgent


def _make_env(config: RLMazeConfig) -> MazeEnv:
    w = 5 if config.effective_tier == "S" else config.maze_width
    h = 5 if config.effective_tier == "S" else config.maze_height
    steps = config.limit_steps(config.max_steps, s_cap=12, m_cap=50)
    return MazeEnv(width=w, height=h, density=config.maze_density, max_steps=steps)


def _make_agent(config: RLMazeConfig, state_size: int):
    if config.agent_type == "q":
        return QAgent(
            lr=config.rl_lr,
            gamma=config.gamma,
            epsilon=config.epsilon,
            epsilon_decay=config.epsilon_decay,
            epsilon_min=config.epsilon_min,
        )
    elif config.agent_type == "dqn":
        return DQNAgent(
            state_size=state_size,
            hidden=config.dqn_hidden,
            lr=config.rl_lr,
            gamma=config.gamma,
            epsilon=config.epsilon,
            epsilon_decay=config.epsilon_decay,
            epsilon_min=config.epsilon_min,
            replay_capacity=config.replay_capacity,
            batch_size=config.replay_batch,
            target_update_every=config.target_update_every,
            device=config.device,
        )
    elif config.agent_type == "ppo":
        return PPOAgent(
            state_size=state_size,
            hidden=config.ppo_hidden,
            lr=config.rl_lr,
            gamma=config.gamma,
            lam=config.ppo_lam,
            clip=config.ppo_clip,
            value_coef=config.ppo_value_coef,
            entropy_coef=config.ppo_entropy_coef,
            ppo_epochs=config.ppo_epochs,
            device=config.device,
        )
    else:
        raise ValueError(f"Unknown agent_type: {config.agent_type!r}")


def _save_agent(config: RLMazeConfig, agent, logger: Logger) -> None:
    """Persist agent weights/table to artifacts directory."""
    if config.agent_type == "q":
        data = {str(k): v for k, v in agent.Q.items()}
        path = logger.artifact_path("agent_q.json")
        with open(path, "w") as f:
            json.dump(data, f)
    elif config.agent_type == "dqn":
        torch.save(agent.online.state_dict(), logger.artifact_path("agent_dqn.pt"))
    elif config.agent_type == "ppo":
        torch.save(agent.ac.state_dict(), logger.artifact_path("agent_ppo.pt"))


class RLMazeTrainer(BaseTrainer):
    """Trains any of the three RL agents on a procedural maze.

    Note: `dataloader` is not used (RL collects its own experience),
    but the signature matches the BaseTrainer contract.
    """

    def __init__(self):
        self.agent = None
        self.env: MazeEnv | None = None

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, RLMazeConfig)
        env = _make_env(config)
        self.env = env
        state = env.reset()
        agent = _make_agent(config, state_size=len(state))
        self.agent = agent

        n_episodes = config.limit_steps(config.n_episodes, s_cap=2, m_cap=20)
        logger.log_config(config.model_dump())

        for ep in range(n_episodes):
            state = env.reset()
            done = False
            ep_reward = 0.0
            trajectory = []

            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state, done)
                trajectory.append((state, action, reward, next_state, done))
                ep_reward += reward
                state = next_state

            agent.end_episode(trajectory)

            success = int(ep_reward > 0)  # positive total reward → reached goal
            logger.log_metrics(ep, {
                "episode_reward": ep_reward,
                "success": success,
                "epsilon": getattr(agent, "epsilon", 0.0),
            })

        _save_agent(config, agent, logger)

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        assert isinstance(config, RLMazeConfig)
        if self.agent is None or self.env is None:
            env = _make_env(config)
            self.env = env
            state = env.reset()
            self.agent = _make_agent(config, len(state))

        env = self.env
        agent = self.agent
        # Set epsilon to 0 for greedy eval (Q and DQN)
        saved_eps = getattr(agent, "epsilon", 0.0)
        if hasattr(agent, "epsilon"):
            agent.epsilon = 0.0

        n_eval = config.limit_steps(config.eval_episodes, s_cap=1, m_cap=5)
        total_reward = 0.0
        successes = 0
        for _ in range(n_eval):
            state = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action = agent.act(state)
                state, reward, done, _ = env.step(action)
                ep_reward += reward
            total_reward += ep_reward
            if ep_reward > 0:
                successes += 1

        if hasattr(agent, "epsilon"):
            agent.epsilon = saved_eps

        return {
            "eval_reward": total_reward / n_eval,
            "success_rate": successes / n_eval,
        }

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        """Run one greedy episode and return the trajectory as a list of actions."""
        assert isinstance(config, RLMazeConfig)
        if self.agent is None or self.env is None:
            raise RuntimeError("Train the agent first.")
        env = self.env
        agent = self.agent

        saved_eps = getattr(agent, "epsilon", 0.0)
        if hasattr(agent, "epsilon"):
            agent.epsilon = 0.0

        state = env.reset()
        done = False
        actions = []
        total_reward = 0.0
        while not done:
            action = agent.act(state)
            actions.append(action)
            state, reward, done, _ = env.step(action)
            total_reward += reward

        if hasattr(agent, "epsilon"):
            agent.epsilon = saved_eps

        return {
            "actions": actions,
            "total_reward": total_reward,
            "success": total_reward > 0,
            "steps": len(actions),
        }

    def load_checkpoint(self, config: BaseConfig, artifacts_dir) -> None:
        """Restore agent from checkpoint. Rebuilds env + agent then loads weights."""
        assert isinstance(config, RLMazeConfig)
        path = Path(artifacts_dir)
        env = _make_env(config)
        self.env = env
        state = env.reset()
        agent = _make_agent(config, len(state))

        if config.agent_type == "q":
            q_path = path / "agent_q.json"
            if q_path.exists():
                with open(q_path) as f:
                    data = json.load(f)
                agent.Q = {ast.literal_eval(k): v for k, v in data.items()}
        elif config.agent_type == "dqn":
            ckpt = path / "agent_dqn.pt"
            if ckpt.exists():
                agent.online.load_state_dict(torch.load(ckpt, map_location=config.device))
                agent.target.load_state_dict(agent.online.state_dict())
                agent.online.eval()
                agent.target.eval()
        elif config.agent_type == "ppo":
            ckpt = path / "agent_ppo.pt"
            if ckpt.exists():
                agent.ac.load_state_dict(torch.load(ckpt, map_location=config.device))
                agent.ac.eval()

        self.agent = agent


def make_rl_maze_dataloader(config: RLMazeConfig, split: str = "train") -> DataLoader:
    """RL doesn't use a dataloader — return a dummy one."""
    from torch.utils.data import TensorDataset
    import torch
    dummy = TensorDataset(torch.zeros(1))
    return DataLoader(dummy, batch_size=1)
