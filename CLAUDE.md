# CLAUDE.md — mini_networks

Educational ML lab: 32 models and 19 cross-model compositions sharing one
runtime contract, one data registry, one logging format, and one quality gate.
Owner-facing reference lab with a graphical **playground** (Observatory).
Nano S-tier runs locally (CPU) for the gate/dev; **M/L training runs on GCP
ephemeral Cloud Run Jobs** (single-train via Pub/Sub, or ONE parallel L4 sweep
job that gates every item as its own task) writing to the **global
`garassino-mlflow` tracker** (Cloud Run + Cloud SQL +
gs://garassino-ml-mlflow-artifacts via --serve-artifacts; deployed from the
desktop garassino-ml workspace — the earlier Neon-DSN plan is superseded).
Gate-passing M/L checkpoints are registered as `mini-<model>` with
champion/challenger promotion. Colab is optional. Design specs:
`docs/superpowers/specs/2026-06-12-ultimate-educational-resource-design.md`
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
  logging/mlflow_registry.py  champion/challenger Model Registry: gate-passing
                     M/L ckpts → `mini-<model>` versions, promoted on the gate
                     metric (MN_MLFLOW_REGISTER=1; never fails a run)
  diffusion/sampling.py  shared DDPM sample_loop (guidance + callbacks)
models/<name>/       config.py + model.py + trainer.py per model
compositions/        multi-model pipelines (each exposes train/sample or compare)
web/                 read-layer: metrics.py (pivot) + sources.py (Local/MLflow/
                     Composite RunSource) + model_catalog.py — reader of the contract
cloud/               publisher.py: JobSpec + Pub/Sub publisher (cloud train dispatch)
                     sweep_shard.py: task-index→item sharding for the parallel
                     sweep job + GCS shard upload + report merge
playground/ (repo root)  Next.js 16 + React 19 + TS + Tailwind v4 + shadcn/ui +
                     Recharts + Motion + Lucide. Toy/storybook "enchanted grove"
                     UI (4 views: Watch/Play/Lab/Quest). Static-exported
                     (output:'export' → playground/out) and served by FastAPI
                     StaticFiles at /. Pure client of /web,/train,/infer. Source
                     committed; out/ + node_modules gitignored (build with
                     `make playground`). Replaced the old no-build vanilla SPA.
infra/gcp/           Dockerfile.train (ARG TORCH_INDEX: cpu|cu121) + entrypoint.sh
                     (MODE=train|sweep-task) + terraform/ (train job, L4 sweep
                     job, topic, trigger, IAM) + function/ (see its README)
colab/
  catalog.py         COMPOSITIONS list, descriptions, categories (MODELS aliases core registry)
  probes.py          per-model inference probes + output validation
  showcase.py        per-item human-viewable inference showcases (grids, text,
                     pred-vs-true, kNN) — saved when args.showcase_dir is set
  runners.py         run_model/run_composition + COMPOSITION_RUNNERS dict
  menu.py            rich TUI + `python -m mini_networks.colab.launcher` CLI
  launcher.py        thin facade keeping the historical import surface
  gate.py            quality-gate runner behind `sweep --check`
api/                 FastAPI: routers/{training,inference,compositions,web}, in-memory
                     jobs; main.py mounts /web read-layer + the SPA at /
main.py (repo root)  argparse CLI: serve|train|evaluate|compose|sweep|
                     sweep-task|sweep-report|menu|list
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
| M | "actually learns" bar (L4 cloud sweep) | ≤5 epochs, 100 batches, batch≤64, 4096 samples, timesteps≤200 (re-sized 2026-07-10; do NOT run full M locally — the owner's machine is slow) |
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
- `make -C infra/gcp build-train[-gpu]` / `dry-run[-sweep]` — build the CPU/cu121 images / run MODE=train or one MODE=sweep-task shard against a local sqlite MLflow
- `make -C infra/gcp sweep TIER=M [ITEMS=a,b,c]` — execute the parallel L4 gate sweep (one Cloud Run task per item); `make -C infra/gcp sweep-report SWEEP=<id>` merges the shards into report.{md,json}
- `make -C infra/gcp sweep-samples SWEEP=<id> [DEST=~/Downloads]` — download every item's inference showcase (sample grids / pred-vs-true / generated text / kNN accuracy; produced by `colab/showcase.py`, uploaded by each sweep task to `…/sweeps/<id>/samples/<item>/`)
- `uv run pytest tests/ -m slow` — slow API tests (full trainings via TestClient)

### Env vars (playground + cloud)

- `MN_MLFLOW_TRACKING_URI` — enable the Logger's MLflow sink. Cloud contract: the global tracker URL `https://garassino-mlflow-mjz4n7eeia-ew.a.run.app` (public; leave `MN_MLFLOW_ARTIFACT_ROOT` unset so artifacts proxy through --serve-artifacts). Local dev may use `sqlite:///…` + `MN_MLFLOW_ARTIFACT_ROOT`.
- `MN_MLFLOW_EXPERIMENT` — experiment name (default `mini-networks`).
- `MN_MLFLOW_REGISTER=1` — gate hook registers gate-passing M/L checkpoints as `mini-<model>` with champion/challenger promotion on the gate metric.
- `MN_RUN_SOURCE` = `local` (default) | `mlflow` — what `/web` reads.
- `MN_TRAIN_BACKEND` = `local` (default) | `cloud` — `POST /train` runs locally vs publishes to Pub/Sub.
- `MN_PUBSUB_TOPIC` / `GOOGLE_CLOUD_PROJECT` — cloud train publisher.
- `MN_SWEEP_BUCKET` / `MN_SWEEP_PREFIX` / `SWEEP_ID` / `ITEMS` / `CLOUD_RUN_TASK_INDEX` — sweep-task sharding + shard upload (bucket unset = local-only dry-run).
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
  S-tier (nano) runs; full sweeps belong to CI (S) or the cloud sweep job (M/L).

## Working state

- Phase 0 (foundation) + Phase 1 (quality gate) + Phase 3 (docs) done.
- **Phase 2 M-tier: 51/51 green (2026-07-10)** via 6 rounds of the parallel
  cloud sweep (`make -C infra/gcp sweep TIER=M` → `sweep-report` → triage;
  rerun failures with `ITEMS=…`). Thresholds carry evidence comments from
  those rounds. CI's `sweep-s` should now be green — remove its
  `continue-on-error` once confirmed.
- Playground + GCP training landed: MLflow sink → global `garassino-mlflow`
  tracker (Cloud SQL; Neon plan superseded), champion/challenger Model
  Registry (`mini-<model>`), `/web` read-layer, Observatory SPA,
  Pub/Sub→Cloud Run Job single-train, ONE parallel L4 sweep job
  (`mini-networks-sweep`, task-sharded gate). Plan B (Sandbox, Lab-compare,
  Lessons, showcase deploy) is future work.
- Known instability fixes applied 2026-07-10: text_seq2seq causal mask +
  shifted teacher forcing + honest eval divisor; pixelcnn Bernoulli BCE +
  true raster-scan sampling; optional `BaseConfig.max_grad_norm` clipping.
