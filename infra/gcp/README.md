# infra/gcp — ephemeral GCP training pipeline

Queue-triggered, scale-to-zero training on Cloud Run Jobs, writing to the
**global garassino MLflow tracker** (`garassino-mlflow` Cloud Run service:
Cloud SQL Postgres backend + `gs://garassino-ml-mlflow-artifacts/` via
`--serve-artifacts`). Nothing runs at rest. Design spec:
`docs/superpowers/specs/2026-06-24-playground-and-gcp-training-design.md`.

Two job shapes share one image family:

```
single train (playground):                       parallel gate sweep:
Lab view / CLI ──▶ Pub/Sub ──▶ Cloud Function    make sweep ──▶ mini-networks-sweep
                                │ runJob            (L4 GPU, N tasks, parallelism 4)
                                ▼                   task i = gate ONE item:
                 mini-networks-train (CPU)          train + checks + EvalSpec + probe
                 MODE=train, one model              MODE=sweep-task
                                │                        │            │ report shard
                                ▼                        ▼            ▼
                 garassino-mlflow (runs + registry)   MLflow    gs://…/mini-networks/
                        ◀── playground reads                    sweeps/<id>/shards/
```

## Files

| Path | Purpose |
|---|---|
| `Dockerfile.train` | training image; `ARG TORCH_INDEX` picks CPU wheels (default) or cu121 for the GPU build |
| `entrypoint.sh` | `MODE=train` (one model) \| `MODE=sweep-task` (one gate item per Cloud Run task) |
| `function/` | Pub/Sub → `runJob` trigger (2nd-gen Cloud Function) |
| `terraform/` | jobs (train, sweep), topic, trigger, IAM, budget |
| `Makefile` | `validate` / `show` / `destroy` / `build-train[-gpu]` / `push-images` / `dry-run[-sweep]` / `sweep` / `sweep-report` |

## Lives elsewhere (not created here)

- GCS bucket `garassino-ml-artifacts` (shared; referenced as a data source).
- The **`garassino-mlflow` tracker** (Cloud Run + Cloud SQL) — deployed from the
  desktop `garassino-ml` workspace; this stack only points at its public URL
  (`var.mlflow_tracking_url`). No secret needed.
- WIF pool + Terraform state bucket in `garassino-op`.

## Env-var contract

**Single train (Pub/Sub message → job execution).** Message:
`{"model","training_tier","epochs","device","run_name","hparams"}`. The trigger
maps it to env overrides `MODEL`, `TRAINING_TIER`, `RUN_NAME`, `DEVICE`,
`EPOCHS`, `HPARAMS`. Static job env (Terraform): `MODE=train`,
`CHECKPOINT_ROOT`, `GOOGLE_CLOUD_PROJECT`, `MN_MLFLOW_TRACKING_URI` (tracker
URL, plain env). `MN_MLFLOW_ARTIFACT_ROOT` stays **unset** so artifacts proxy
through the tracker's `--serve-artifacts` root.

**Parallel sweep (`make sweep` → execution overrides).** `ITEMS` (comma list,
catalog order — the task-index → item mapping both sides derive from
`cloud/sweep_shard.default_items()`), `SWEEP_ID`, `TRAINING_TIER`; `--tasks`
matches `ITEMS`. Static: `MODE=sweep-task`, `DEVICE=cuda`, `MN_SWEEP_BUCKET`,
`MN_SWEEP_PREFIX`, `MN_MLFLOW_REGISTER=1` (gate-passing M/L checkpoints are
registered as `mini-<model>` and champion/challenger-promoted on the gate
metric). GCS auth is ADC via the runtime service account — no JSON keys.

## Status / completion

There is **no callback** to the API. Jobs write run status to MLflow
(`RUNNING → FINISHED/FAILED`); sweep tasks additionally upload a JSON shard per
item. `gcloud run jobs executions describe` is an ops-only secondary signal.

## Verify (no cloud spend)

```bash
make validate       # terraform fmt -check + validate
make build-train    # docker build the CPU training image
make dry-run        # MODE=train against a local sqlite MLflow, assert exit 0 + FINISHED run
make dry-run-sweep  # MODE=sweep-task shard 0 (classifier) locally, shard JSON under /tmp
```

## Show / run / destroy (deliberate, cost-approved)

```bash
make push-images    # build + push CPU and cu121 images to GAR ml-images
make show           # terraform apply

# smoke: 2 tasks at S tier
make sweep TIER=S ITEMS=classifier,gan
make sweep-report SWEEP=<id> ITEMS=classifier,gan

# the real thing: full 51-item M sweep on L4 tasks (~1-2h wall, ~€4-7)
make sweep TIER=M
make sweep-report SWEEP=<id>

make destroy        # tear down; images stay in Artifact Registry
```

Single-train path unchanged:
```bash
gcloud pubsub topics publish mini-networks-train-requests \
  --message '{"model":"classifier","training_tier":"S","run_name":"smoke-1"}'
```

## Gotchas

- **Jobs pin image digests at deploy time.** Pushing a new `:latest` (locally or
  via `gcloud builds submit --config infra/gcp/cloudbuild.yaml .`) does NOT
  reach existing jobs — run `make refresh-jobs` after every image push.
- **`.gcloudignore` patterns must stay anchored** (leading `/`): an unanchored
  `data/` once excluded `src/mini_networks/core/data/` and shipped a broken image.
- **L4 quota bounds parallelism** (terraform `sweep_parallelism`, default 3 =
  the region's `NvidiaL4GpuAllocNoZonalRedundancyPerProjectRegion` quota).
  Tasks beyond it queue — safe, just slower. The GPU job shape (gen2 +
  `nvidia-l4` node selector + zonal redundancy off) mirrors the live
  `ppe-train` job, which proves europe-west1 GPU jobs work.
- **ITEMS order is load-bearing.** The task index maps into the ITEMS list;
  `make sweep` always passes ITEMS explicitly so container and caller agree.
- **Trigger needs `roles/run.developer`** (not `run.invoker`) for `runJob` with
  overrides, plus `actAs` on the runtime SA.
- **A task that dies before uploading its shard** shows up in `sweep-report`
  as `missing_shards` and fails the merge — rerun just those items with
  `make sweep ITEMS=a,b,c`.
