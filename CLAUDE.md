# CLAUDE.md — mini_networks

Educational ML lab: ~31 models and ~19 cross-model compositions sharing one
runtime contract, one data registry, one logging format, and one quality gate.
Owner-facing reference lab with a graphical **playground** (Observatory).
Nano S-tier runs locally (CPU) for the gate/dev; **M/L training runs on GCP
ephemeral Cloud Run Jobs** (queue-triggered, scale-to-zero) writing to a
persistent MLflow contract (Neon Postgres + GCS); Colab is now optional. Design
specs: `docs/superpowers/specs/2026-06-12-ultimate-educational-resource-design.md`
and `…/2026-06-24-playground-and-gcp-training-design.md`.

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
  logging/mlflow_sink.py  optional env-gated MLflow mirror (MN_MLFLOW_TRACKING_URI);
                     Logger dual-writes; ended by Logger.close() + atexit net
  diffusion/sampling.py  shared DDPM sample_loop (guidance + callbacks)
models/<name>/       config.py + model.py + trainer.py per model
compositions/        multi-model pipelines (each exposes train/sample or compare)
web/                 read-layer: metrics.py (pivot) + sources.py (Local/MLflow/
                     Composite RunSource) + model_catalog.py — reader of the contract
cloud/               publisher.py: JobSpec + Pub/Sub publisher (cloud train dispatch)
playground/ (repo root)  Next.js 16 + React 19 + TS + Tailwind v4 + shadcn/ui +
                     Recharts + Motion + Lucide. Toy/storybook "enchanted grove"
                     UI (4 views: Watch/Play/Lab/Quest). Static-exported
                     (output:'export' → playground/out) and served by FastAPI
                     StaticFiles at /. Pure client of /web,/train,/infer. Source
                     committed; out/ + node_modules gitignored (build with
                     `make playground`). Replaced the old no-build vanilla SPA.
infra/gcp/           Dockerfile.train + entrypoint.sh (MODE=train) + terraform/ +
                     function/ — ephemeral Cloud Run Job pipeline (see its README)
colab/
  catalog.py         COMPOSITIONS list, descriptions, categories (MODELS aliases core registry)
  probes.py          per-model inference probes + output validation
  runners.py         run_model/run_composition + COMPOSITION_RUNNERS dict
  menu.py            rich TUI + `python -m mini_networks.colab.launcher` CLI
  launcher.py        thin facade keeping the historical import surface
  gate.py            quality-gate runner behind `sweep --check`
api/                 FastAPI: routers/{training,inference,compositions,web}, in-memory
                     jobs; main.py mounts /web read-layer + the SPA at /
main.py (repo root)  argparse CLI: serve|train|evaluate|compose|sweep|menu|list
```

Read-layer principle: the playground is a pure **reader of the run contract**.
Trainers never change — they call `Logger`, which (when `MN_MLFLOW_TRACKING_URI`
is set) also mirrors to MLflow. `MN_RUN_SOURCE` (local|mlflow) selects what the
`/web` endpoints read; `MN_TRAIN_BACKEND` (local|cloud) selects whether
`POST /train` runs in-process or publishes to Pub/Sub.

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
- `python main.py serve` — FastAPI on :8000; playground at `/` (serves `playground/out`), API docs at `/docs`
- `make playground` — build the Next.js UI to `playground/out` (rebuild after UI changes; CI builds it for deploy)
- `make playground-dev` — Next dev server on :3000 with hot reload (proxies API to :8000 via `NEXT_PUBLIC_API_BASE`)
- `make -C infra/gcp validate` — terraform fmt-check + validate (static, no cloud)
- `make -C infra/gcp build-train` / `dry-run` — build the train image / run MODE=train against a local sqlite MLflow
- `uv run pytest tests/ -m slow` — slow API tests (full trainings via TestClient)

### Env vars (playground + cloud)

- `MN_MLFLOW_TRACKING_URI` / `MN_MLFLOW_ARTIFACT_ROOT` / `MN_MLFLOW_EXPERIMENT` — enable + configure the Logger's MLflow sink.
- `MN_RUN_SOURCE` = `local` (default) | `mlflow` — what `/web` reads.
- `MN_TRAIN_BACKEND` = `local` (default) | `cloud` — `POST /train` runs locally vs publishes to Pub/Sub.
- `MN_PUBSUB_TOPIC` / `GOOGLE_CLOUD_PROJECT` — cloud train publisher.
- Cloud/MLflow deps live in the `cloud` extra (`uv sync --extra cloud`); all imports are lazy so the base install + `smoke-import` stay light.

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

- Phase 0 (foundation) + Phase 1 (quality gate) + Phase 3 (docs) done.
- Phase 2: M-tier stabilization on Colab GPU until all 50 items pass
  (driver: `colab/notebooks/00_sweep.ipynb`).
- Playground + GCP-ephemeral training (Plan A) landed: MLflow sink, `/web`
  read-layer, Observatory SPA, Pub/Sub→Cloud Run Job pipeline (infra static-
  validated; live apply is a deliberate cost-approved step). Plan B (Sandbox,
  Lab-compare, Lessons, showcase deploy) is future work.
