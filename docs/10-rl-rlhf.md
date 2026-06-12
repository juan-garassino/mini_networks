# Chapter 10 — Reinforcement Learning and RLHF

## Theory recap

RL agents learn from reward instead of labels. **Q-learning** keeps a table of
state-action values updated toward `r + γ·max_a Q(s', a)`. **DQN** replaces
the table with a neural network, stabilized by an experience-replay buffer
(breaks temporal correlation) and a frozen target network (fixed Q-targets).
**REINFORCE** skips values entirely and pushes up log-probabilities of actions
in proportion to the episode return. **PPO** is the modern policy-gradient
workhorse: it limits each update by clipping the probability ratio between new
and old policies. **RLHF** applies PPO to a language model — generate text,
score it with a reward signal, and update the policy while a KL penalty
against a frozen reference model prevents reward hacking.

## In this repo

### RL maze (`src/mini_networks/models/rl_maze/`)

- `env.py` — `MazeEnv`, a dependency-free procedural grid world. State is the
  flattened 5×5 neighbourhood plus the agent's (row, col) → 27-dim vector.
  4 actions (up/down/left/right); rewards: +1.0 goal, −0.5 hole (terminal),
  −0.01 per step. `render()` prints ASCII.
- `agents.py` — three agents behind one interface (`act`, `update`,
  `end_episode`): `QAgent` (tabular, quantized state keys, epsilon-greedy),
  `DQNAgent` (MLP Q-network + replay buffer + target network synced every
  `target_update_every` episodes), and `PPOAgent` (actor-critic with GAE
  λ=0.95, clipped surrogate, entropy bonus, batch update per episode).
- `trainer.py` — `RLMazeTrainer` selects via `config.agent_type` (`"q"` |
  `"dqn"` | `"ppo"`, default `"dqn"`) and saves per-agent checkpoint formats:
  `agent_q.json` (the Q-table), `agent_dqn.pt` (online network weights), or
  `agent_ppo.pt` (actor-critic weights). `load_checkpoint()` rebuilds the
  env + agent from whichever file is present.

### REINFORCE (`src/mini_networks/models/reinforce/`)

- `model.py` — `PolicyNet`, a two-layer MLP returning `log_softmax` over
  actions. `trainer.py` runs full episodes on the same `MazeEnv`, computes
  discounted returns back-to-front, normalizes them, and minimizes
  `-(log_probs * returns).sum()` — the classic score-function estimator.
  No dataloader is needed; `make_reinforce_dataloader` returns a dummy.

### RLHF (`src/mini_networks/models/rlhf/`)

- `trainer.py` — two stages. Stage 1 pretrains a `TransformerLM` on the text
  corpus with cross-entropy. Stage 2 freezes a `copy.deepcopy` of it as the
  reference policy, then loops: sample prompts from the corpus → generate
  rollouts → score each with the heuristic `shakespearean_score` reward
  (`model.py`) → PPO update. Per rollout, the token-level KL penalty is
  literally `kl = (log_probs_ref - log_probs_cur).mean()`, and the policy
  loss uses ratio clipping against the frozen reference:
  `ratio = exp(log_probs_cur - log_probs_ref.detach())`, then
  `-min(ratio * advantage, clamp(ratio, 1-clip, 1+clip) * advantage)`.
  Total loss is `policy_loss + kl_coef * kl`. The episode's scalar reward is
  broadcast to every token — a simplification of full token-level credit
  assignment that keeps the PPO mechanics intact.

## Composition: RL + RLHF in one task domain

`src/mini_networks/compositions/rl_rlhf_maze.py` bridges both paradigms on
the same maze. Phase 1: solve the maze with DQN. Phase 2: encode DQN
trajectories as character strings ("SUURRDDG" = start, moves, goal;
"X" = failure) over an 8-token action alphabet — `_VOCAB = "SUDLRXG"` plus
PAD (`VOCAB_SIZE = 8`) — and pretrain a small `TransformerLM` on that corpus.
Phase 3: fine-tune the LM with PPO where the reward is the maze itself
(decode the generated action string, replay it in the env, reward 0/1) —
RLHF with an objective reward instead of human labels. `compare()` returns
`{dqn_success_rate, lm_success_rate, dqn_mean_steps, lm_mean_steps,
sample_trajectories}` so you can see whether language modelling actually
learned to navigate.

## Try it

```bash
uv run python main.py train --model rl_maze --fast_demo
uv run python main.py train --model reinforce --fast_demo
uv run python main.py train --model rlhf --fast_demo
```

## Latest results

<!-- results:start items=rl_maze,reinforce,rlhf -->

_Latest sweep: tier S on cpu_

| Item | Status | Metric | Value | Threshold |
|---|---|---|---|---|
| rl_maze | pass | success_rate | 0.0000 | n/a |
| reinforce | pass | success_rate | 0.0000 | n/a |
| rlhf | pass | eval_loss | 3.9089 | n/a |

<!-- results:end -->
