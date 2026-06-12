# CLAUDE.md — mini_networks

Educational ML lab: ~31 models and ~19 cross-model compositions sharing one
runtime contract, one data registry, one logging format, and one quality gate.
Owner-facing reference lab; trainings run via CLI locally (CPU) or notebooks on
a Colab GPU kernel. Design spec:
`docs/superpowers/specs/2026-06-12-ultimate-educational-resource-design.md`.

## Architecture (src/mini_networks/)

```
core/
  config.py          BaseConfig (pydantic v2) + tier system (see below)
  runtime.py         BaseTrainer ABC: train/evaluate/infer/load_checkpoint
                     + SupervisedTrainer / ContrastiveTrainer / reconstruction bases
  evalspec.py        EvalSpec + EVAL_SPECS — quality-gate thresholds, 1 entry per item
  sweep_report.py    CheckResult + report.{md,json} writers (stdlib-only)
  checkpoints.py     latest_run_dir / find_resumable_run
  data/registry.py   get_dataset/get_dataloader — MNIST task modes, text, audio, tabular
  logging/logger.py  Logger → runs/<name>/<ts>/{metrics.jsonl,config.yaml,artifacts/}
  diffusion/sampling.py  shared DDPM sample_loop (guidance + callbacks)
models/<name>/       config.py + model.py + trainer.py per model
compositions/        multi-model pipelines (each exposes train/sample or compare)
colab/
  catalog.py         COMPOSITIONS list, descriptions, categories (MODELS aliases core registry)
  probes.py          per-model inference probes + output validation
  runners.py         run_model/run_composition + COMPOSITION_RUNNERS dict
  menu.py            rich TUI + `python -m mini_networks.colab.launcher` CLI
  launcher.py        thin facade keeping the historical import surface
  gate.py            quality-gate runner behind `sweep --check`
api/                 FastAPI: routers/{training,inference,compositions}, in-memory jobs
main.py (repo root)  argparse CLI: serve|train|evaluate|compose|sweep|menu|list
```

Registry: `core/registry.py::get_model_registry()` (cached) maps
`name → (ConfigClass, TrainerClass, dataloader_fn)`; `MODEL_NAMES` is the
static name list (sync-tested). Compositions dispatch through
`runners.COMPOSITION_RUNNERS` (sync-tested against the catalog).
Supervised vision trainers (classifier/resnet/vit/mobilenet/convnext)
implement only `_build`; `_forward`/`infer` live on `SupervisedTrainer`,
and they share `core/data/registry.py::make_classification_dataloader`.

## Tier system (BaseConfig)

| Tier | Means | Budgets |
|---|---|---|
| S | nano smoke run (CPU, CI) | 1 epoch, 1 train batch, batch≤16, 32 samples, 1 eval batch, diffusion timesteps capped to 25 |
| M | "actually learns" bar (Colab GPU) | ≤3 epochs, 8 batches, 512 samples, timesteps≤200 |
| L | full budget | config values as-is |

`--fast_demo` forces S. Use `config.effective_*` properties (epochs, batch_size,
timesteps, …) — never raw fields — so every code path honors the tier.
`config.effective_timesteps` must be used for BOTH scheduler construction and
sampling so the noise chain stays consistent.

## Quality gate

`python main.py sweep --check [--training_tier S|M|L]` — trains every selected
item, then per item: finiteness + loss-trend S-checks from metrics.jsonl,
checkpoint round-trip through a fresh trainer (`load_checkpoint`), EvalSpec
metric on the loaded weights (threshold gates at M/L only), inference probe.
Writes `runs/sweep/<ts>/report.{md,json}`; non-zero exit on any non-pass.

- Thresholds live in `core/evalspec.py`. **Threshold honesty rule:** changing a
  number requires a justification comment next to it.
- diffusion/gan/pixelcnn are scored by a trained MNIST classifier
  (`gate.judge_samples`: confidence × class coverage); the gate trains or
  reuses the judge automatically.
- Single-item rerun: `--models <name> --skip-compositions`.

## Commands

- `uv sync` / `uv sync --dev` — install
- `make test` — fast suite (slow marker deselected via addopts)
- `make validate-s` — full S-tier check sweep (what CI runs)
- `python main.py train --model <name> --fast_demo` — one nano training
- `python main.py sweep --check --fast_demo --models clip,gan --skip-compositions` — targeted gate
- `python main.py serve` — FastAPI on :8000
- `uv run pytest tests/ -m slow` — slow API tests (full trainings via TestClient)

## CI (.github/workflows/ci.yml)

Jobs: `smoke-import` (imports every module), `test` (`make test-ci`), `sweep-s`
(full S-tier check sweep, report uploaded as artifact). `sweep-s` is
`continue-on-error: true` until Phase 2 stabilization makes it green — remove
that line when it does.

## Conventions

- Python 3.11, uv-managed, torch pinned `>=2.1,<2.3` + numpy `<2` (old x86_64 mac)
- `from __future__ import annotations` everywhere; argparse only; stdlib logging
  (`log = logging.getLogger(__name__)` — `logger` is reserved for the metrics Logger
  parameter in trainers)
- `torch.load` always with `weights_only=True`
- Tests: pytest, flat `tests/`, stub/skip pattern for download-gated datasets
  (`tests/conftest.py::dataset_or_skip`)
- `legacy/` is reference-only; its datasets/venvs are untracked (history was
  rewritten 2026-06-12 to purge binaries — do not re-add binary data to git)
- The owner's machine is slow: locally run only targeted tests and single
  S-tier (nano) runs; full sweeps belong to CI or Colab.

## Working state

- Phase 0 (foundation) + Phase 1 (quality gate) done.
- Phase 2: M-tier stabilization on Colab GPU until all 50 items pass
  (driver: `colab/notebooks/00_sweep.ipynb`).
- Phase 3: docs/ curriculum chapters + annotated model sources.
