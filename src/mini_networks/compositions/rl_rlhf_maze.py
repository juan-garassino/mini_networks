"""RL + RLHF Maze composition.

Educational goal
----------------
  Bridges two paradigms: classic RL (DQN on a grid maze) and RLHF (PPO
  fine-tuning of a language model) using the same task domain.

  Key insight: encode maze actions as single chars so the maze becomes a
  language modelling problem.  We can then:
    1. Solve the maze with DQN (standard RL).
    2. Collect DQN trajectories and use them to pretrain a small Transformer.
    3. Fine-tune the Transformer with PPO using the maze reward (0/1 for
       goal/failure) instead of human labels — this is RLHF with an
       objective reward.

Action encoding
---------------
  0 (UP)    → "U"
  1 (DOWN)  → "D"
  2 (LEFT)  → "L"
  3 (RIGHT) → "R"

Trajectory format
-----------------
  "SUURRDDG\\n"  — successful episode (goal reached)
  "SUURDX\\n"    — failed episode (fell into hole or time-out = "X")

  Each trajectory is one line in the corpus.
  The LM learns to generate valid-looking maze paths starting from "S".

Three-phase pipeline
--------------------
  Phase 1 — Train DQNAgent on MazeEnv (standard RL).
  Phase 2 — Collect N trajectories, encode as char corpus,
             pretrain TransformerLM with next-token CE loss.
  Phase 3 — RLHF: for each rollout sample an action sequence from the LM,
             execute in MazeEnv, get binary reward (1=goal, 0=other),
             apply PPO with KL penalty vs frozen reference model.
"""
from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from pydantic import Field

from mini_networks.core.config import BaseConfig
from mini_networks.core.logging.logger import Logger
from mini_networks.models.rl_maze.agents import DQNAgent
from mini_networks.models.rl_maze.env import MazeEnv
from mini_networks.models.transformer.model import TransformerLM


# ---------------------------------------------------------------------------
# Action ↔ char encoding
# ---------------------------------------------------------------------------

_ACTION_TO_CHAR = {0: "U", 1: "D", 2: "L", 3: "R"}
_CHAR_TO_ACTION = {v: k for k, v in _ACTION_TO_CHAR.items()}
_VOCAB = "SUDLRXG"   # 7 chars + we add PAD implicitly
_STOI  = {c: i + 1 for i, c in enumerate(_VOCAB)}  # 1-indexed; 0 = PAD
_ITOS  = {v: k for k, v in _STOI.items()}
VOCAB_SIZE = len(_STOI) + 1   # 8


def _encode(trajectory: str) -> list[int]:
    return [_STOI.get(c, 0) for c in trajectory]


def _decode(ids: list[int]) -> str:
    return "".join(_ITOS.get(i, "") for i in ids if i != 0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class RLHFMazeConfig(BaseConfig):
    model_name: str = "rl_rlhf_maze"

    # Maze
    maze_width:  int = 5
    maze_height: int = 5
    maze_density: float = 0.15
    max_steps: int = 50

    # DQN (Phase 1)
    rl_agent_type: str = "dqn"
    rl_episodes: int = 200          # fast_demo: 30
    dqn_hidden: int = 64
    rl_lr: float = 1e-3
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.99
    epsilon_min: float = 0.05

    # Trajectory collection
    n_collect: int = 200            # episodes to collect after DQN training

    # LM (Phase 2 – pretrain)
    lm_d_model: int = 64
    lm_n_layers: int = 2
    lm_n_heads: int = 2
    lm_d_ff: int = 128
    lm_seq_len: int = 64
    lm_pretrain_epochs: int = 3     # fast_demo: 1

    # RLHF PPO (Phase 3)
    n_rollouts: int = 20            # fast_demo: 4
    rollout_max_steps: int = 50     # max tokens LM may generate per rollout
    ppo_epochs: int = 4             # fast_demo: 1
    rlhf_lr: float = 3e-4
    kl_coef: float = 0.01
    clip_eps: float = 0.2

    # Evaluation
    eval_episodes: int = 20         # fast_demo: 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_maze(config: RLHFMazeConfig) -> MazeEnv:
    w = 5 if config.fast_demo else config.maze_width
    h = 5 if config.fast_demo else config.maze_height
    steps = 30 if config.fast_demo else config.max_steps
    return MazeEnv(width=w, height=h, density=config.maze_density, max_steps=steps)


def _make_dqn(config: RLHFMazeConfig, state_size: int) -> DQNAgent:
    return DQNAgent(
        state_size=state_size,
        hidden=config.dqn_hidden,
        lr=config.rl_lr,
        gamma=config.gamma,
        epsilon=config.epsilon,
        epsilon_decay=config.epsilon_decay,
        epsilon_min=config.epsilon_min,
        replay_capacity=1000,
        batch_size=32,
        target_update_every=50,
        device=config.device,
    )


def _build_lm(config: RLHFMazeConfig) -> TransformerLM:
    return TransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=config.lm_d_model,
        n_heads=config.lm_n_heads,
        n_layers=config.lm_n_layers,
        d_ff=config.lm_d_ff,
        seq_len=config.lm_seq_len,
        dropout=0.1,
    ).to(config.device)


def _run_episode(env: MazeEnv, agent: DQNAgent, train: bool = True) -> tuple[list[int], float]:
    """Run one episode. Returns (actions, total_reward)."""
    state = env.reset()
    done = False
    actions: list[int] = []
    total_reward = 0.0
    trajectory: list = []

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        if train:
            agent.update(state, action, reward, next_state, done)
        trajectory.append((state, action, reward, next_state, done))
        actions.append(action)
        total_reward += reward
        state = next_state

    if train:
        agent.end_episode(trajectory)

    return actions, total_reward


def _actions_to_trajectory_str(actions: list[int], success: bool) -> str:
    """Encode actions as char string: S + action chars + G or X."""
    body = "".join(_ACTION_TO_CHAR.get(a, "U") for a in actions)
    return "S" + body + ("G" if success else "X")


def _log_probs_of_tokens(
    model: TransformerLM,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Log P(token[t] | token[:t]) for all t>0. Shape [T-1]."""
    logits, _ = model(tokens)              # [1, T, V]
    logits = logits[0, :-1, :]            # [T-1, V]
    targets = tokens[0, 1:]               # [T-1]
    return -F.cross_entropy(logits, targets, reduction="none")


# ---------------------------------------------------------------------------
# Main composition class
# ---------------------------------------------------------------------------

class RLHFMazeComposition:
    """Three-phase pipeline: DQN → pretrain LM → RLHF fine-tune.

    Usage::

        comp   = RLHFMazeComposition()
        logger = Logger("runs/rl_rlhf_maze/001")
        comp.train(config, logger)
        result = comp.compare(config)
    """

    def __init__(self):
        self.dqn_agent: DQNAgent | None = None
        self.env: MazeEnv | None = None
        self.lm: TransformerLM | None = None
        self.ref_lm: TransformerLM | None = None

    # ------------------------------------------------------------------
    # Phase 1: train DQN
    # ------------------------------------------------------------------

    def _phase1_train_dqn(
        self,
        config: RLHFMazeConfig,
        logger: Logger,
    ) -> DQNAgent:
        env = _make_maze(config)
        self.env = env
        state = env.reset()
        agent = _make_dqn(config, len(state))

        n_ep = 30 if config.fast_demo else config.rl_episodes
        successes = 0
        for ep in range(n_ep):
            actions, ep_reward = _run_episode(env, agent, train=True)
            success = ep_reward > 0
            if success:
                successes += 1
            logger.log_metrics(ep, {
                "phase": 1,
                "episode_reward": ep_reward,
                "success": int(success),
                "epsilon": agent.epsilon,
            })

        print(f"  [Phase 1] DQN done — success rate: {successes / max(1, n_ep):.2%}")
        self.dqn_agent = agent
        return agent

    # ------------------------------------------------------------------
    # Phase 2: collect trajectories → pretrain LM
    # ------------------------------------------------------------------

    def _phase2_pretrain_lm(
        self,
        config: RLHFMazeConfig,
        agent: DQNAgent,
        logger: Logger,
    ) -> TransformerLM:
        env = self.env or _make_maze(config)

        # Collect trajectories (mix of successes + failures)
        n_collect = 50 if config.fast_demo else config.n_collect
        saved_eps = agent.epsilon
        agent.epsilon = 0.2  # some exploration for diversity
        trajectories: list[str] = []
        for _ in range(n_collect):
            actions, reward = _run_episode(env, agent, train=False)
            traj_str = _actions_to_trajectory_str(actions, reward > 0)
            trajectories.append(traj_str)
        agent.epsilon = saved_eps

        corpus = "\n".join(trajectories)
        print(f"  [Phase 2] Corpus: {len(trajectories)} trajectories, "
              f"{len(corpus)} chars")

        # Build simple token sequences from corpus
        all_ids: list[int] = _encode(corpus)

        model = _build_lm(config)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        epochs = 1 if config.fast_demo else config.lm_pretrain_epochs
        seq_len = min(config.lm_seq_len, 32)

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0
            # Create sliding-window chunks
            for i in range(0, max(1, len(all_ids) - seq_len), seq_len):
                chunk = all_ids[i: i + seq_len + 1]
                if len(chunk) < 2:
                    continue
                x = torch.tensor([chunk[:-1]], dtype=torch.long, device=config.device)
                y = torch.tensor([chunk[1:]], dtype=torch.long, device=config.device)
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg = total_loss / max(1, n_batches)
            logger.log_metrics(config.rl_episodes + epoch, {
                "phase": 2,
                "pretrain_loss": avg,
            })
            print(f"  [Phase 2] epoch {epoch}  loss {avg:.4f}")

        self.lm = model
        return model

    # ------------------------------------------------------------------
    # Phase 3: RLHF PPO fine-tuning
    # ------------------------------------------------------------------

    def _phase3_rlhf(
        self,
        config: RLHFMazeConfig,
        model: TransformerLM,
        logger: Logger,
    ) -> None:
        env = self.env or _make_maze(config)

        # Freeze reference policy
        ref = copy.deepcopy(model)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        self.ref_lm = ref

        optimizer = optim.AdamW(model.parameters(), lr=config.rlhf_lr)
        n_rollouts = 4 if config.fast_demo else config.n_rollouts
        ppo_epochs = 1 if config.fast_demo else config.ppo_epochs
        offset = config.rl_episodes + config.lm_pretrain_epochs

        # Start token id
        start_id = _STOI.get("S", 1)

        for it in range(n_rollouts):
            # --- Collect rollout: sample from LM, execute in maze ---
            model.eval()
            rollouts: list[dict] = []
            max_gen = min(config.rollout_max_steps, 30 if config.fast_demo else 50)

            prompt = torch.tensor([[start_id]], dtype=torch.long, device=config.device)
            with torch.no_grad():
                generated = model.generate(prompt, max_new_tokens=max_gen, temperature=1.0)
            gen_ids = generated[0].tolist()
            decoded = _decode(gen_ids)

            # Execute decoded actions in environment
            env_state = env.reset()
            done = False
            steps_taken = 0
            maze_reward = 0.0
            for char in decoded[1:]:  # skip leading "S"
                if char in ("G", "X", ""):
                    break
                action = _CHAR_TO_ACTION.get(char, 0)
                env_state, r, done, _ = env.step(action)
                maze_reward += r
                steps_taken += 1
                if done:
                    break

            # Binary reward: 1.0 = goal reached, 0.0 otherwise
            reward = 1.0 if maze_reward > 0 else 0.0
            rollouts.append({"tokens": generated, "reward": reward})

            # --- PPO update ---
            model.train()
            total_loss = 0.0
            for _ in range(ppo_epochs):
                for rb in rollouts:
                    tokens = rb["tokens"]  # [1, T]
                    if tokens.shape[1] < 2:
                        continue
                    cur_lp = _log_probs_of_tokens(model, tokens)
                    with torch.no_grad():
                        ref_lp = _log_probs_of_tokens(ref, tokens)

                    kl = (ref_lp - cur_lp).mean()
                    r_t = torch.tensor(rb["reward"], dtype=torch.float32, device=config.device)
                    advantage = r_t - 0.5  # centre

                    ratio = torch.exp(cur_lp - ref_lp.detach())
                    clipped = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
                    policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
                    loss = policy_loss + config.kl_coef * kl

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()

            avg_loss = total_loss / max(1, len(rollouts) * ppo_epochs)
            avg_reward = sum(rb["reward"] for rb in rollouts) / max(1, len(rollouts))
            logger.log_metrics(offset + it, {
                "phase": 3,
                "ppo_loss": avg_loss,
                "avg_reward": avg_reward,
            })
            print(f"  [Phase 3] iter {it}  loss {avg_loss:.4f}  reward {avg_reward:.3f}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, config: RLHFMazeConfig, logger: Logger) -> None:
        """Run all three phases."""
        logger.log_config(config.model_dump())
        print("[RLHFMaze] Phase 1: training DQN agent")
        agent = self._phase1_train_dqn(config, logger)
        print("[RLHFMaze] Phase 2: pretraining LM on maze trajectories")
        lm = self._phase2_pretrain_lm(config, agent, logger)
        print("[RLHFMaze] Phase 3: RLHF PPO fine-tuning")
        self._phase3_rlhf(config, lm, logger)
        print("[RLHFMaze] Done.")

    def compare(self, config: RLHFMazeConfig, n_eval: int | None = None) -> dict:
        """Evaluate DQN agent vs fine-tuned LM on the maze.

        Returns::

            {
                "dqn_success_rate": float,
                "lm_success_rate":  float,
                "dqn_mean_steps":   float,
                "lm_mean_steps":    float,
                "sample_trajectories": list[str],
            }
        """
        if self.dqn_agent is None or self.lm is None:
            raise RuntimeError("Call train() first.")

        env = self.env or _make_maze(config)
        n = n_eval or (5 if config.fast_demo else config.eval_episodes)
        start_id = _STOI.get("S", 1)

        # Evaluate DQN (greedy, epsilon=0)
        dqn_agent = self.dqn_agent
        saved_eps = dqn_agent.epsilon
        dqn_agent.epsilon = 0.0
        dqn_successes = 0
        dqn_steps_total = 0
        for _ in range(n):
            actions, reward = _run_episode(env, dqn_agent, train=False)
            dqn_successes += int(reward > 0)
            dqn_steps_total += len(actions)
        dqn_agent.epsilon = saved_eps

        # Evaluate LM (sample + execute)
        lm = self.lm
        lm.eval()
        lm_successes = 0
        lm_steps_total = 0
        sample_trajectories: list[str] = []
        max_gen = min(config.rollout_max_steps, 30 if config.fast_demo else 50)

        with torch.no_grad():
            for _ in range(n):
                prompt = torch.tensor([[start_id]], dtype=torch.long, device=config.device)
                gen = lm.generate(prompt, max_new_tokens=max_gen, temperature=0.8)
                decoded = _decode(gen[0].tolist())
                sample_trajectories.append(decoded[:40])

                env_state = env.reset()
                done = False
                steps = 0
                maze_reward = 0.0
                for char in decoded[1:]:
                    if char in ("G", "X", ""):
                        break
                    action = _CHAR_TO_ACTION.get(char, 0)
                    env_state, r, done, _ = env.step(action)
                    maze_reward += r
                    steps += 1
                    if done:
                        break
                lm_successes += int(maze_reward > 0)
                lm_steps_total += steps

        return {
            "dqn_success_rate": dqn_successes / max(1, n),
            "lm_success_rate":  lm_successes  / max(1, n),
            "dqn_mean_steps":   dqn_steps_total / max(1, n),
            "lm_mean_steps":    lm_steps_total  / max(1, n),
            "sample_trajectories": sample_trajectories[:5],
        }


# ---------------------------------------------------------------------------
# Convenience launcher (mirrors run_composition in launcher.py)
# ---------------------------------------------------------------------------

def run(config: RLHFMazeConfig | None = None, logger: Logger | None = None) -> dict:
    """Train the composition and return compare() results."""
    import os
    import time
    if config is None:
        config = RLHFMazeConfig(fast_demo=True)
    if logger is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = os.path.join(config.output_dir, f"rl_rlhf_maze/{ts}")
        logger = Logger(out)
    comp = RLHFMazeComposition()
    comp.train(config, logger)
    return comp.compare(config)
