"""REINFORCE trainer for MazeEnv."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.logging.logger import Logger
from mini_networks.core.runtime import BaseTrainer
from mini_networks.models.reinforce.config import ReinforceConfig
from mini_networks.models.reinforce.model import PolicyNet
from mini_networks.models.rl_maze.env import MazeEnv


def _make_env(config: ReinforceConfig) -> MazeEnv:
    w = 5 if config.fast_demo else config.maze_width
    h = 5 if config.fast_demo else config.maze_height
    steps = 50 if config.fast_demo else config.max_steps
    return MazeEnv(width=w, height=h, density=config.maze_density, max_steps=steps)


class ReinforceTrainer(BaseTrainer):
    def __init__(self):
        self.policy: PolicyNet | None = None
        self.env: MazeEnv | None = None

    def _build(self, config: ReinforceConfig, state_size: int) -> PolicyNet:
        return PolicyNet(state_size=state_size, hidden_dim=config.hidden_dim).to(config.device)

    def train(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> None:
        assert isinstance(config, ReinforceConfig)
        env = _make_env(config)
        self.env = env
        state = env.reset()
        policy = self._build(config, state_size=len(state))
        self.policy = policy
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        logger.log_config(config.model_dump())

        n_episodes = 10 if config.fast_demo else config.n_episodes
        for ep in range(n_episodes):
            state = env.reset()
            done = False
            log_probs = []
            rewards = []
            ep_reward = 0.0

            while not done:
                s = torch.tensor(state, dtype=torch.float32, device=config.device).unsqueeze(0)
                logp = policy(s).squeeze(0)
                action = int(torch.distributions.Categorical(logits=logp).sample().item())
                next_state, reward, done, _ = env.step(action)
                log_probs.append(logp[action])
                rewards.append(reward)
                ep_reward += reward
                state = next_state

            # Compute returns
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + config.gamma * G
                returns.append(G)
            returns = list(reversed(returns))
            returns = torch.tensor(returns, dtype=torch.float32, device=config.device)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            loss = -torch.stack(log_probs) * returns
            loss = loss.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            success = int(ep_reward > 0)
            logger.log_metrics(ep, {"episode_reward": ep_reward, "success": success})

        torch.save(policy.state_dict(), logger.artifact_path("model.pt"))

    def evaluate(self, config: BaseConfig, dataloader: DataLoader, logger: Logger) -> dict:
        assert isinstance(config, ReinforceConfig)
        if self.policy is None or self.env is None:
            env = _make_env(config)
            self.env = env
            state = env.reset()
            self.policy = self._build(config, state_size=len(state))

        env = self.env
        policy = self.policy
        policy.eval()

        n_eval = 5 if config.fast_demo else config.eval_episodes
        total_reward = 0.0
        successes = 0
        with torch.no_grad():
            for _ in range(n_eval):
                state = env.reset()
                done = False
                ep_reward = 0.0
                while not done:
                    s = torch.tensor(state, dtype=torch.float32, device=config.device).unsqueeze(0)
                    logp = policy(s).squeeze(0)
                    action = int(torch.argmax(logp).item())
                    state, reward, done, _ = env.step(action)
                    ep_reward += reward
                total_reward += ep_reward
                if ep_reward > 0:
                    successes += 1

        return {
            "eval_reward": total_reward / n_eval,
            "success_rate": successes / n_eval,
        }

    def infer(self, config: BaseConfig, inputs: Any) -> Any:
        assert isinstance(config, ReinforceConfig)
        if self.policy is None or self.env is None:
            raise RuntimeError("Train the agent first.")
        env = self.env
        policy = self.policy
        policy.eval()

        state = env.reset()
        done = False
        actions = []
        total_reward = 0.0
        with torch.no_grad():
            while not done:
                s = torch.tensor(state, dtype=torch.float32, device=config.device).unsqueeze(0)
                logp = policy(s).squeeze(0)
                action = int(torch.argmax(logp).item())
                actions.append(action)
                state, reward, done, _ = env.step(action)
                total_reward += reward
        return {
            "actions": actions,
            "total_reward": total_reward,
            "success": total_reward > 0,
            "steps": len(actions),
        }


def make_reinforce_dataloader(config: ReinforceConfig, split: str = "train") -> DataLoader:
    """REINFORCE doesn't use a dataloader — return a dummy one."""
    from torch.utils.data import TensorDataset
    dummy = TensorDataset(torch.zeros(1))
    return DataLoader(dummy, batch_size=1)
