"""Three RL agents for MazeEnv: Q-table, DQN, PPO.

All agents share a simple interface:
  act(state)                    → int action
  update(state, action, reward, next_state, done) → optional float loss
  end_episode(trajectory)       → for PPO batch update (ignored by others)

Educational comparison
----------------------
  QAgent   — tabular, no neural network, can overfit small mazes perfectly
  DQNAgent — neural Q-network + replay buffer + target network
             handles large state spaces but may be brittle
  PPOAgent — on-policy actor-critic with clipped surrogate + GAE
             most stable on continuous/complex tasks
"""
from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Q-table agent
# ---------------------------------------------------------------------------

class QAgent:
    """Tabular Q-learning with epsilon-greedy exploration.

    State keys are tuples (quantized to 1 decimal place to reduce explosion).
    """

    def __init__(
        self,
        n_actions: int = 4,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q: dict = defaultdict(float)

    def _key(self, state: np.ndarray) -> tuple:
        # Quantize to reduce Q-table size
        return tuple(np.round(state, 1).tolist())

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        k = self._key(state)
        q_vals = [self.Q[(k, a)] for a in range(self.n_actions)]
        return int(np.argmax(q_vals))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        k = self._key(state)
        nk = self._key(next_state)
        current_q = self.Q[(k, action)]
        if done:
            target = reward
        else:
            best_next = max(self.Q[(nk, a)] for a in range(self.n_actions))
            target = reward + self.gamma * best_next
        self.Q[(k, action)] = current_q + self.lr * (target - current_q)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return abs(target - current_q)

    def end_episode(self, trajectory) -> None:
        pass  # Q-table updates online; nothing to do


# ---------------------------------------------------------------------------
# DQN agent
# ---------------------------------------------------------------------------

class _QNetwork(nn.Module):
    def __init__(self, state_size: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """Deep Q-Network with experience replay and a target network.

    Educational notes:
      - Replay buffer breaks temporal correlations → more stable training.
      - Target network provides fixed Q-targets for a window → less oscillation.
      - Epsilon decays over episodes for exploration → exploitation transition.
    """

    def __init__(
        self,
        state_size: int,
        n_actions: int = 4,
        hidden: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        replay_capacity: int = 1000,
        batch_size: int = 32,
        target_update_every: int = 100,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.device = device
        self._episodes = 0

        self.online = _QNetwork(state_size, n_actions, hidden).to(device)
        self.target = _QNetwork(state_size, n_actions, hidden).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer: deque = deque(maxlen=replay_capacity)

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(s).argmax(dim=1).item())

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        self.buffer.append((state, action, reward, next_state, float(done)))
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s  = torch.tensor(np.array(states),      dtype=torch.float32, device=self.device)
        a  = torch.tensor(actions,                dtype=torch.long,    device=self.device)
        r  = torch.tensor(rewards,                dtype=torch.float32, device=self.device)
        ns = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        d  = torch.tensor(dones,                  dtype=torch.float32, device=self.device)

        current_q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(ns).max(1)[0]
        target_q = r + (1 - d) * self.gamma * next_q

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def end_episode(self, trajectory) -> None:
        self._episodes += 1
        if self._episodes % self.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())


# ---------------------------------------------------------------------------
# PPO agent
# ---------------------------------------------------------------------------

class _ActorCritic(nn.Module):
    def __init__(self, state_size: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor):
        return F.softmax(self.actor(x), dim=-1), self.critic(x).squeeze(-1)


class PPOAgent:
    """Proximal Policy Optimization with Generalized Advantage Estimation.

    Educational notes:
      - On-policy: collects a full episode trajectory before updating.
      - Clipped surrogate objective prevents destructive large updates.
      - GAE (λ=0.95) balances bias vs variance in advantage estimation.
      - Entropy bonus encourages exploration by rewarding uncertain policies.
    """

    def __init__(
        self,
        state_size: int,
        n_actions: int = 4,
        hidden: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        ppo_epochs: int = 4,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.device = device
        # PPO doesn't use epsilon; expose for API compatibility
        self.epsilon = 0.0

        self.ac = _ActorCritic(state_size, n_actions, hidden).to(device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

    def act(self, state: np.ndarray) -> int:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            probs, _ = self.ac(s)
            return int(Categorical(probs).sample().item())

    def update(self, state, action, reward, next_state, done) -> float:
        # PPO updates in end_episode; online update is no-op
        return 0.0

    def end_episode(self, trajectory) -> float:
        """Batch update from a full episode trajectory.

        trajectory: list of (state, action, reward, next_state, done)
        """
        if not trajectory:
            return 0.0

        states, actions, rewards, next_states, dones = zip(*trajectory)
        s  = torch.tensor(np.array(states),      dtype=torch.float32, device=self.device)
        a  = torch.tensor(actions,                dtype=torch.long,    device=self.device)
        r  = torch.tensor(rewards,                dtype=torch.float32, device=self.device)
        ns = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        d  = torch.tensor(dones,                  dtype=torch.float32, device=self.device)

        # Compute GAE advantages and value targets
        with torch.no_grad():
            _, values      = self.ac(s)
            _, next_values = self.ac(ns)

        deltas = r + self.gamma * next_values * (1 - d) - values
        advantages = self._gae(deltas.cpu().numpy(), dones)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + values.detach()

        # Normalise advantages (guard against single-step trajectories)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # Get old log-probs (fixed reference)
        with torch.no_grad():
            old_probs, _ = self.ac(s)
            old_log_probs = Categorical(old_probs).log_prob(a)

        total_loss = 0.0
        for _ in range(self.ppo_epochs):
            probs, vals = self.ac(s)
            dist = Categorical(probs)
            log_probs = dist.log_prob(a)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(vals, returns)
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / self.ppo_epochs

    def _gae(self, deltas: np.ndarray, dones) -> np.ndarray:
        """Generalized Advantage Estimation."""
        gae = 0.0
        advantages = np.zeros_like(deltas)
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.gamma * self.lam * (1 - float(dones[t])) * gae
            advantages[t] = gae
        return advantages
