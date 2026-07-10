# mini_networks — Playground UI + GCP Ephemeral Training (design)

Date: 2026-06-24
Status: design — awaiting review
Supersedes/extends: the training-execution story in
`2026-06-12-ultimate-educational-resource-design.md` (which assumed local CPU +
Colab GPU). This document keeps that lab intact and adds two things on top: a
graphical **playground** and a **GCP-ephemeral training substrate** with
persistent tracking.

---

## 1. Problem & motivation

The lab today is *usable* (CLI, FastAPI, rich TUI) but not *seeable*. A learner
can run `python main.py train --model vae` and get `runs/<id>/metrics.jsonl` +
artifacts on disk, but never **watches a network learn** or **pokes a trained
model live**. For an owner-facing reference lab that is acceptable; for an
"incredible educational playground" the observable, interactive experience *is*
the value.

Separately, real ("actually learns", M/L tier) training currently leans on Colab
GPU notebooks. The owner wants training to run on **GCP infrastructure**,
triggered through a **message queue**, on **ephemeral compute** (deploy image →
train → kill), with **persistent experiment tracking and models** — so no
container is up all the time.

These two needs share one resolution: make the thing the UI reads a **persistent
contract** that ephemeral GCP jobs write to.

## 2. Goals / non-goals

**Goals**
- A single graphical playground with four modes: Observatory, Sandbox, Lab,
  Lessons — serving three audiences from one codebase (owner / learners /
  public showcase).
- M/L training runs on GCP, queue-triggered, ephemeral, scale-to-zero.
- Experiment metadata + model artifacts persist independently of compute.
- Trainers remain UI- and infra-agnostic (the existing clean architecture).
- Stay inside the €2–10/mo cost target; nothing always-on except (optionally,
  later) a min-instances=0 frontend.

**Non-goals**
- Replacing the CLI/TUI/quality-gate. They stay; the S-tier nano gate keeps
  running in CI against local `runs/` files.
- A heavyweight frontend toolchain (no npm/React build step).
- Multi-GPU / distributed training (the models are deliberately tiny).
- An always-on MLflow tracking server.

## 3. Core principle (load-bearing)

**The UI is a reader of a persistent contract; trainers only write to a Logger.**

- Today's contract: `runs/<id>/{metrics.jsonl, config.yaml, artifacts/}`.
- New contract for GCP runs: **MLflow stores** (Postgres metadata + GCS
  artifacts). MLflow is an *additional sink*, env-gated; local runs keep writing
  files so the quality gate is unaffected.
- A live run is just a contract still being appended — so the same reader code
  gives **live and replay** for free.

This is why the design is small: nothing in `models/*/trainer.py` changes. The
seam is `core/logging/logger.py`.

## 4. Architecture overview

```
[Frontend · Lab view]  or  [CLI]
        │  publish {model, tier, hparams}
        ▼
   Pub/Sub topic            (durable, retry, nothing running)
        │  Eventarc push
        ▼
   Trigger · Cloud Function  (serverless)
        │  run execution
        ▼
   Ephemeral training job · Cloud Run Job
   (image from GAR · entrypoint MODE=train · CPU default, GPU opt-in ·
    scale-to-zero · self-terminates)
        │ params+metrics            │ model+samples
        ▼                            ▼
   MLflow store · Neon Postgres   GCS · gs://garassino-ml-artifacts/mini-networks
        └──────── the persistent contract (survives every kill) ────────┘
                          ▲  read (MlflowClient on DB URI + GCS), on demand
   Frontend reader · 4 views
   (local `python main.py serve` by default; deployable to garassino-ai later)
```

Cross-project placement (per root CLAUDE.md GCP architecture):
- **garassino-ml** — Pub/Sub topic, trigger function, Cloud Run training jobs,
  GCS artifacts. Ephemeral; no always-on workload.
- **garassino-ai** — optional frontend (Cloud Run, `min-instances=0`). Deferred.
- **garassino-op** — WIF pool for CI/deploy, Neon DSN in Secret Manager.
- Region `europe-west1`. No service-account JSON keys (WIF only).

## 5. Decisions locked (this session)

| Decision | Choice | Why |
|---|---|---|
| Audience | All three, tiered | One substrate, framing differs per tier |
| Core experiences | All four views | Share one shell + the read-layer |
| Delivery | Read-layer SPA (FastAPI-served) | Reuses API + contract; trainers untouched |
| Training compute | **Cloud Run Jobs** (CPU default, L4 GPU opt-in) | Scale-to-zero, cheapest; tiny models don't need Vertex; GCE-GPU only as escalation |
| Tracking backend | **MLflow → Neon Postgres + GCS** | Neon precedent; CLAUDE.md bans Cloud SQL; no server needed with DB URI |
| Frontend tech | **No-build vanilla** ES modules + tiny chart (uPlot/SVG), FastAPI static | Python-only repo, runs on Colab, readable, matches "no Node in a Python project" |
| Frontend hosting | **Local-first, deploy later** | Same binary local vs Cloud Run; defer the showcase flip |

## 6. Sub-project A — GCP training pipeline

**A.1 Training image.** Reuse the RunPod-style `entrypoint.sh` with a `MODE=`
switch (`train` to start). Image built and pushed to GAR (or GHCR for the public
showcase variant) by CI. Reads a job spec (model, tier, hyperparams) from env/args.

**A.2 MLflow logging sink.** Extend `core/logging/logger.py` with an optional
MLflow backend, activated by env (e.g. `MN_MLFLOW_TRACKING_URI`,
`MN_MLFLOW_ARTIFACT_ROOT`). When set, `log_metric/log_metrics/log_config/
log_summary/artifact_path` also write to MLflow (DB URI → direct to Postgres;
artifacts → GCS). When unset, behaviour is exactly today's file logging. No
trainer changes.

**A.3 Queue + trigger.** A Pub/Sub topic receives `{model, tier, hparams,
run_name}`. An Eventarc-triggered Cloud Function launches a Cloud Run Job
execution with those as env. The job self-terminates on completion (Cloud Run
Jobs are ephemeral by construction).

**A.4 Persistence.** Neon free-tier Postgres = MLflow backend store (DSN from
`op/` Secret Manager, never committed). GCS `gs://garassino-ml-artifacts/
mini-networks/` = artifact root. Models are MB-scale → storage cost ≈ €0.

**A.5 Compute routing.** Default CPU Cloud Run Job. Per-model GPU opt-in (L4)
for the heavy few (diffusion, transformer, vit) where the region supports
Cloud Run GPU; otherwise those escalate to an ephemeral GCE L4 VM (self-deleting,
same entrypoint). Most of the 31 models run CPU.

**A.6 What stays local / CI.** The S-tier nano quality gate (`sweep --check`)
stays in CI against local `runs/` — fast, free, unchanged. Local dev runs still
write `runs/<id>/` files. GCP path is for M/L "real" runs that the owner wants
observable + persistent.

**A.7 Security / IaC.** WIF through garassino-op for CI image push and job
launch. Terraform (show-and-destroy) for the topic, function, job, and IAM
bindings. Budget alerts already per CLAUDE.md.

## 7. Sub-project B — Playground frontend

**B.1 Tech.** No-build: plain ES-module JS + one tiny chart approach
(uPlot ~40KB or hand-rolled SVG), served as static files by the existing
FastAPI app. No bundler, no Node. Clone-and-run; works identically on Colab.

**B.2 Backend read-layer** (new `src/mini_networks/web/`, mounted on the
existing app):
- `GET /runs`, `GET /runs/{id}/metrics?since=N`, `/runs/{id}/artifacts/...`,
  `/runs/{id}/config`, `/runs/{id}/gate` — backed by **either** local `runs/`
  **or** MLflow (`MlflowClient` on the DB URI + GCS), selected by env. Same
  response schemas either way.
- `GET /models` — registry introspection (uses `core/registry.py`) + per-model
  probe metadata for the Sandbox.
- Extend existing `POST /infer/{model}` for Sandbox interactions.
- `POST /train` (Lab): local mode → existing in-memory runner; cloud mode →
  publish to Pub/Sub.
- `GET /lessons` — serves `docs/00..11`, each mapped to a model/run.

**B.3 Views (one shell, tabs).**
- **Observatory** (Phase 1) — run picker + live loss curves + sample grid +
  config/gate panel. Pure reader; no new model code.
- **Sandbox** — per-model input widgets (draw-a-digit, latent slider, prompt)
  over `/infer`.
- **Lab** — launch form (model, tier, hparams) → cloud trigger; run-compare table.
- **Lessons** — curriculum chapters with inline runnable embeds.

**B.4 Hosting.** Default: `python main.py serve` opens the playground locally,
reading remote MLflow. Optional later flip: deploy the same app to garassino-ai
Cloud Run (`min-instances=0`, ~€0 idle) for the public showcase, DSN from Secret
Manager via WIF.

## 8. Persistence / data contract

| Concept | Local file contract | MLflow contract |
|---|---|---|
| config | `config.yaml` | `log_params` (Postgres) |
| metrics | `metrics.jsonl` | `log_metric` per step (Postgres) |
| samples/ckpts | `artifacts/` | GCS under run's artifact root |
| summary/gate | `log_summary` | run tags / metrics |

The read-layer normalizes both into the same Pydantic response models, so the
frontend is identical regardless of source.

## 9. Phasing

1. **Observatory MVP** — MLflow logging sink (A.2) + read-layer + SPA shell +
   Observatory view. Validate against a manually-launched GCP job (minimal
   trigger) or local MLflow. *You see a net learn, persisted.*
2. **GCP trigger pipeline** — Pub/Sub topic + Cloud Function + Cloud Run Job +
   self-kill + Terraform/WIF. Lab "launch" becomes a real cloud trigger.
3. **Sandbox** — inference widgets per model family.
4. **Lab compare + Lessons** — run-compare table; interactive curriculum.
5. **Optional showcase deploy** — frontend to garassino-ai (min=0), GPU opt-in
   routing for heavy models.

Implementation proceeds as **two sequenced plans**: Plan A (Phases 1–2, the
pipeline + Observatory) then Plan B (Phases 3–5, the richer views + deploy).

## 10. Testing

- pytest over the read-layer against a fixture `runs/` dir and a fake/local
  MLflow (sqlite backend + tmp artifact dir) — same assertions both sources.
- Schema tests on new Pydantic response models.
- The existing `smoke-import` CI job covers the new `web/` package.
- MLflow sink tested by logging to a sqlite+tmp MLflow and asserting reads.
- Trigger/job: a unit test for the message → job-spec mapping; the Terraform is
  statically validated (the project's convention for infra not yet live-run).
- Frontend kept thin enough to verify via endpoint contracts; no JS test
  harness introduced.

## 11. Risks & mitigations

- **Owner's slow local machine** — Observatory is read-only (fine locally); live
  fidelity happens on the GCP job, not the laptop.
- **Cloud Run GPU regional availability** — default CPU; GPU opt-in only where
  supported; GCE L4 escalation otherwise. Verify europe-west1 at implementation.
- **Neon DSN exposure** — never committed; Secret Manager via WIF when deployed;
  local dev uses a local sqlite MLflow.
- **Scope** — mitigated by shipping Phase 1 (Observatory) standalone.
- **Cost drift** — Pub/Sub free tier, per-execution jobs on tiny models, Neon
  €0, GCS cents; budget alerts already configured.

## 12. Docs to update (same commits as the code)

- **CLAUDE.md** (project) — new `web/` package + read-layer; `serve` now opens
  the playground; training story updated to "GCP-ephemeral canonical for M/L,
  local nano for gate/dev, Colab optional"; MLflow sink env vars.
- **README.md** — how to launch the playground; how to trigger a GCP run.
- **AGENTS.md / DOCS.md** — if they enumerate surfaces, add the playground +
  GCP pipeline.
- **Root CLAUDE.md GCP section** — note mini-networks now uses garassino-ml
  ephemeral jobs + (later) a garassino-ai min=0 frontend.

## 13. Open questions (minor, not blocking)

- Chart lib: uPlot vs hand-rolled SVG — decide in Plan B by bundle-size feel.
- Whether the public showcase image lives in GHCR (public) or GAR (private) —
  defaults to GHCR per the registry-by-visibility rule; confirm at Phase 5.
