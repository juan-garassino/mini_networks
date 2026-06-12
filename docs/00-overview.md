# 00 — Overview: how this repo works

`mini_networks` is an educational ML framework: ~30 small models and ~19 multi-model
compositions, all built on one runtime contract, one data registry, one logger, and one
quality gate. Every chapter in `docs/` assumes you have read this one.

## Repo map

```
main.py                          # single CLI entry point (argparse)
src/mini_networks/
  core/                          # the framework
    config.py                    #   BaseConfig (Pydantic) + tier-aware effective_* properties
    runtime.py                   #   BaseTrainer ABC + Supervised/Contrastive/Seg/Det bases
    tiers.py                     #   S/M/L budget table
    evalspec.py                  #   per-model quality thresholds
    data/registry.py             #   dataset factory (see docs/01-data.md)
    logging/logger.py            #   unified run-directory logger
    blocks/, diffusion/          #   shared nn building blocks, shared DDPM sampling
  models/<name>/                 # one folder per model: config.py, model.py, trainer.py
  compositions/                  # multi-model pipelines (latent_diffusion, clip_guided_diffusion, ...)
  api/                           # FastAPI server (POST /train/{model}, /infer/{model})
  colab/                         # catalog, probes, runners, menu (TUI), gate.py (quality gate)
colab/notebooks/                 # 00_sweep, 01_vision, 02_language, 03_rl, 04_compositions
tests/                           # fast pytest suite (slow API tests opt-in)
scripts/render_results.py        # injects sweep results into these docs
runs/                            # all training output lands here
```

## The runtime contract

Every model implements `BaseTrainer` (`src/mini_networks/core/runtime.py`):

| Method | Signature | Job |
|---|---|---|
| `train` | `(config, dataloader, logger) -> None` | run the loop, log metrics, save weights to `artifacts/` |
| `evaluate` | `(config, dataloader, logger) -> dict` | return a metrics dict (e.g. `{"accuracy": ...}`) |
| `infer` | `(config, inputs) -> Any` | one forward pass / sampling call on arbitrary inputs |
| `load_checkpoint` | `(config, artifacts_dir) -> None` | rebuild via `self._build(config)` and load `model.pt`; overridden by GAN/diffusion/RL/text trainers |

Most classifiers just subclass `SupervisedTrainer` and implement `_build` (model
construction) plus optionally `_forward`/`_loss`. The base class owns the Adam loop,
accuracy tracking, resume-from-`training_state.pt`, and checkpoint saving.

## Logger output layout

`Logger` (`src/mini_networks/core/logging/logger.py`) writes one directory per run:

```
runs/<name>/<timestamp>/
  metrics.jsonl        # one JSON line per (step, key, value)
  config.yaml          # full config dump at train start
  summary.json         # final status
  training_state.pt    # resumable model+optimizer state
  artifacts/           # model.pt, samples, tokenizer.json, ...
```

Nothing writes anywhere else. `trainer.evaluate` reads nothing from disk; `evaluate`
from a saved run goes through `load_checkpoint` on the `artifacts/` directory.

## Tiers: S / M / L

Budgets live in one visible table, `src/mini_networks/core/tiers.py`. `BaseConfig`
reads them through `effective_epochs`, `effective_batch_size`, `max_train_batches`,
`max_eval_batches`, `dataset_sample_limit`, and `effective_timesteps`.

| Tier | Meaning | Default budget |
|---|---|---|
| **S** | smoke run — "does it execute end-to-end" | 1 epoch, 1 train batch, 1 eval batch, batch cap 16, 32 samples, 25 diffusion timesteps |
| **M** | GPU "actually learns" bar — must clear its EvalSpec threshold | 3 epochs, 8 train / 4 eval batches, batch cap 32, 512 samples, 200 timesteps |
| **L** | full training, uncapped | everything `None` (uncapped) |

`--fast_demo` forces tier S. Per-model exceptions go in `MODEL_OVERRIDES` in
`core/tiers.py`, each with a comment saying why.

## The quality gate

`src/mini_networks/core/evalspec.py` holds one `EvalSpec` per model and composition:
a metric name, `{"M": x, "L": y}` thresholds, direction, and the loss keys used for
S-tier trend checks. Threshold changes require a justification comment next to the
number. At S tier the gate only checks: training completes, all losses are finite,
and the loss trend points down. At M/L the metric must clear the threshold.

Run the gate:

```bash
uv run python main.py sweep --check                      # all models + compositions
uv run python main.py sweep --check --models classifier,vit --skip-compositions
```

Per item it trains at the requested tier, runs the S-style sanity checks on
`metrics.jsonl`, round-trips the checkpoint through a *fresh* trainer, evaluates the
EvalSpec metric on the loaded weights, and probes inference (`colab/gate.py`). The
report lands in `runs/sweep/<timestamp>/report.{md,json}` and the process exits
non-zero if anything fails.

## How to run anything

```bash
uv sync                                                  # install
uv run python main.py list                               # all models & compositions
uv run python main.py train --model resnet --training_tier M
uv run python main.py compose --composition latent_diffusion --fast_demo
uv run python main.py sweep --check                      # the quality gate
uv run python main.py serve                              # FastAPI on :8000
uv run python main.py evaluate --model vae --checkpoint runs/vae/<ts>/artifacts
```

On Colab, open `colab/notebooks/00_sweep.ipynb` (the gate as a notebook), or run
`main.py` with no sub-command — it auto-detects Colab and opens the interactive menu.
After a check sweep, `uv run python scripts/render_results.py` fills the results
blocks at the bottom of each docs chapter from the latest `report.json`.

## Latest results

<!-- results:start items=classifier,diffusion,transformer -->

_Latest sweep: tier S on cpu_

| Item | Status | Metric | Value | Threshold |
|---|---|---|---|---|
| classifier | pass | accuracy | 0.0000 | n/a |
| diffusion | pass | judge_score | 0.0223 | n/a |
| transformer | pass | eval_loss | 3.2333 | n/a |

<!-- results:end -->
