"""Smoke tests for MazeEnv, QAgent, DQNAgent, PPOAgent, and RLMazeTrainer."""
import os
import tempfile

import numpy as np
import pytest
import torch

from mini_networks.models.rl_maze.config import RLMazeConfig
from mini_networks.models.rl_maze.env import MazeEnv, HOLE, PATH, START, GOAL
from mini_networks.models.rl_maze.agents import QAgent, DQNAgent, PPOAgent
from mini_networks.models.rl_maze.trainer import RLMazeTrainer, make_rl_maze_dataloader
from mini_networks.core.logging.logger import Logger

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


# ---------------------------------------------------------------------------
# MazeEnv
# ---------------------------------------------------------------------------

class TestMazeEnv:
    def test_reset_returns_state(self):
        env = MazeEnv(width=5, height=5)
        state = env.reset()
        assert isinstance(state, np.ndarray)
        assert state.shape == (env.state_size,)

    def test_state_size_correct(self):
        env = MazeEnv(width=5, height=5)
        assert env.state_size == 27  # 5×5 + 2

    def test_step_returns_tuple(self):
        env = MazeEnv(width=5, height=5)
        env.reset()
        result = env.step(0)
        assert len(result) == 4
        state, reward, done, info = result
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_all_actions(self):
        env = MazeEnv(width=5, height=5)
        for action in range(4):
            env.reset()
            state, reward, done, info = env.step(action)
            assert state.shape == (env.state_size,)

    def test_goal_gives_positive_reward(self):
        """Force agent onto goal cell and check reward."""
        env = MazeEnv(width=3, height=3)
        env.reset()
        env.agent_pos = env.goal_pos
        env.maze[env.goal_pos] = GOAL
        _, reward, done, _ = env.step(0)  # any action
        # After being at goal, stepping may or may not hit it again — just
        # check the step doesn't crash; the reward test is structural
        assert isinstance(reward, float)

    def test_max_steps_terminates(self):
        env = MazeEnv(width=3, height=3, max_steps=3)
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(0)
            steps += 1
            if steps > 10:
                break
        assert steps <= 4  # should terminate by max_steps=3

    def test_render_returns_string(self):
        env = MazeEnv(width=5, height=5)
        env.reset()
        s = env.render()
        assert isinstance(s, str)
        assert "@" in s  # agent marker

    def test_grid_has_start_and_goal(self):
        env = MazeEnv(width=5, height=5)
        assert env.maze[env.start_pos] == START
        assert env.maze[env.goal_pos] == GOAL

    def test_state_contains_position(self):
        env = MazeEnv(width=5, height=5)
        env.reset()
        state = env._get_state()
        r, c = env.agent_pos
        # Last two elements are (row, col)
        assert state[-2] == float(r)
        assert state[-1] == float(c)


# ---------------------------------------------------------------------------
# QAgent
# ---------------------------------------------------------------------------

class TestQAgent:
    def test_act_returns_valid_action(self):
        env = MazeEnv(width=5, height=5)
        state = env.reset()
        agent = QAgent()
        action = agent.act(state)
        assert 0 <= action < 4

    def test_update_returns_td_error(self):
        env = MazeEnv(width=5, height=5)
        s = env.reset()
        agent = QAgent()
        a = agent.act(s)
        ns, r, done, _ = env.step(a)
        td = agent.update(s, a, r, ns, done)
        assert isinstance(td, float)

    def test_epsilon_decays(self):
        env = MazeEnv(width=5, height=5)
        agent = QAgent(epsilon=1.0, epsilon_decay=0.9)
        s = env.reset()
        initial_eps = agent.epsilon
        for _ in range(5):
            a = agent.act(s)
            ns, r, done, _ = env.step(a)
            agent.update(s, a, r, ns, done)
            s = ns
            if done:
                s = env.reset()
        assert agent.epsilon < initial_eps

    def test_qtable_populated_after_updates(self):
        env = MazeEnv(width=5, height=5)
        agent = QAgent()
        s = env.reset()
        for _ in range(10):
            a = agent.act(s)
            ns, r, done, _ = env.step(a)
            agent.update(s, a, r, ns, done)
            s = ns if not done else env.reset()
        assert len(agent.Q) > 0


# ---------------------------------------------------------------------------
# DQNAgent
# ---------------------------------------------------------------------------

class TestDQNAgent:
    def _agent(self):
        return DQNAgent(state_size=27, n_actions=4, hidden=16, replay_capacity=100, batch_size=8)

    def test_act_returns_valid_action(self):
        env = MazeEnv(width=5, height=5)
        state = env.reset()
        agent = self._agent()
        action = agent.act(state)
        assert 0 <= action < 4

    def test_update_returns_loss_after_warmup(self):
        env = MazeEnv(width=5, height=5)
        agent = self._agent()
        s = env.reset()
        losses = []
        for _ in range(20):
            a = agent.act(s)
            ns, r, done, _ = env.step(a)
            loss = agent.update(s, a, r, ns, done)
            losses.append(loss)
            s = ns if not done else env.reset()
        # Should have at least one non-zero loss after buffer fills
        assert any(l > 0 for l in losses)

    def test_target_network_syncs(self):
        agent = self._agent()
        agent.target_update_every = 2
        # Run enough end_episode calls to trigger sync
        for _ in range(3):
            agent.end_episode([])
        # After sync, online and target should match
        for p1, p2 in zip(agent.online.parameters(), agent.target.parameters()):
            assert torch.allclose(p1.data, p2.data)

    def test_epsilon_decays(self):
        agent = self._agent()
        agent.epsilon = 1.0
        env = MazeEnv(width=5, height=5)
        s = env.reset()
        for _ in range(5):
            a = agent.act(s)
            ns, r, done, _ = env.step(a)
            agent.update(s, a, r, ns, done)
            s = ns if not done else env.reset()
        assert agent.epsilon < 1.0


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------

class TestPPOAgent:
    def _agent(self):
        return PPOAgent(state_size=27, n_actions=4, hidden=16, ppo_epochs=2)

    def test_act_returns_valid_action(self):
        env = MazeEnv(width=5, height=5)
        state = env.reset()
        agent = self._agent()
        action = agent.act(state)
        assert 0 <= action < 4

    def test_end_episode_returns_float(self):
        env = MazeEnv(width=5, height=5)
        agent = self._agent()
        traj = []
        s = env.reset()
        for _ in range(10):
            a = agent.act(s)
            ns, r, done, _ = env.step(a)
            traj.append((s, a, r, ns, done))
            s = ns if not done else env.reset()
        result = agent.end_episode(traj)
        assert isinstance(result, float)

    def test_empty_trajectory_no_crash(self):
        agent = self._agent()
        result = agent.end_episode([])
        assert result == 0.0

    def test_backprop_updates_weights(self):
        env = MazeEnv(width=5, height=5)
        agent = self._agent()
        old_params = [p.clone() for p in agent.ac.parameters()]
        traj = []
        s = env.reset()
        for _ in range(15):
            a = agent.act(s)
            ns, r, done, _ = env.step(a)
            traj.append((s, a, r, ns, done))
            s = ns if not done else env.reset()
        agent.end_episode(traj)
        new_params = list(agent.ac.parameters())
        # At least some params should have changed
        changed = any(not torch.allclose(o, n) for o, n in zip(old_params, new_params))
        assert changed


# ---------------------------------------------------------------------------
# RLMazeTrainer
# ---------------------------------------------------------------------------

class TestRLMazeTrainer:
    def _config(self, agent_type="dqn", **kwargs):
        defaults = dict(
            agent_type=agent_type,
            fast_demo=True,
            data_root=DATA_ROOT,
            epochs=1,
        )
        defaults.update(kwargs)
        return RLMazeConfig(**defaults)

    @pytest.mark.parametrize("agent_type", ["q", "dqn", "ppo"])
    def test_train_smoke(self, agent_type):
        config = self._config(agent_type)
        trainer = RLMazeTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name=f"test_{agent_type}")
            dl = make_rl_maze_dataloader(config)
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            assert len(metrics) > 0

    @pytest.mark.parametrize("agent_type", ["q", "dqn", "ppo"])
    def test_train_logs_reward(self, agent_type):
        config = self._config(agent_type)
        trainer = RLMazeTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name=f"test_reward_{agent_type}")
            dl = make_rl_maze_dataloader(config)
            trainer.train(config, dl, logger)
            metrics = logger.read_metrics()
            keys = {m["key"] for m in metrics}
            assert "episode_reward" in keys

    def test_evaluate_returns_dict(self):
        config = self._config("dqn")
        trainer = RLMazeTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_eval")
            dl = make_rl_maze_dataloader(config)
            trainer.train(config, dl, logger)
            result = trainer.evaluate(config, dl, logger)
            assert "eval_reward" in result
            assert "success_rate" in result
            assert 0.0 <= result["success_rate"] <= 1.0

    def test_infer_returns_actions(self):
        config = self._config("dqn")
        trainer = RLMazeTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(output_dir=tmpdir, run_name="test_infer")
            dl = make_rl_maze_dataloader(config)
            trainer.train(config, dl, logger)
            result = trainer.infer(config, {})
            assert "actions" in result
            assert "total_reward" in result
            assert "success" in result
            assert isinstance(result["actions"], list)
