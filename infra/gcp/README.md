# infra/gcp — ephemeral GCP training pipeline

Queue-triggered, scale-to-zero training on Cloud Run Jobs, writing to a persistent
MLflow contract (Neon Postgres + GCS). Nothing runs at rest. Design spec:
`docs/superpowers/specs/2026-06-24-playground-and-gcp-training-design.md`.

```
Lab view / CLI ──publish──▶ Pub/Sub topic ──▶ Cloud Function (trigger)
                                                   │ runJob(overrides)
                                                   ▼
                                  Cloud Run Job (entrypoint MODE=train)
                                   │ params+metrics        │ model+samples
                                   ▼                        ▼
                            Neon Postgres            GCS mini-networks/artifacts
                                   └──── the persistent contract ────┘ ◀── playground reads
```

## Files

| Path | Purpose |
|---|---|
| `Dockerfile.train` | training image (base + `cloud` extra); sibling to the light serve image |
| `entrypoint.sh` | `MODE=train` one-shot: runs the train CLI, exits cleanly |
| `function/` | Pub/Sub → `runJob` trigger (2nd-gen Cloud Function) |
| `terraform/` | job(s), topic, trigger, IAM, secret, budget |
| `Makefile` | `validate` / `show` / `destroy` / `build-train` / `dry-run` |

## Lives elsewhere (not created here)

- GCS bucket `garassino-ml-artifacts` (shared; referenced as a data source).
- WIF pool + Terraform state bucket in `garassino-op`.
- The **Neon DSN value** — added out-of-band to Secret Manager (never in git/state).

## Env-var contract (Pub/Sub message → job execution)

Message: `{"model","training_tier","epochs","device","run_name","hparams"}`.
The trigger maps it to job env overrides: `MODEL`, `TRAINING_TIER`, `RUN_NAME`,
`DEVICE`, `EPOCHS`, `HPARAMS`. Static job env (Terraform) supplies `MODE=train`,
`CHECKPOINT_ROOT`, `MN_MLFLOW_ARTIFACT_ROOT`, `GOOGLE_CLOUD_PROJECT`, and
`MN_MLFLOW_TRACKING_URI` (Neon DSN from Secret Manager). GCS/Neon auth is ADC via
the runtime service account — no JSON keys.

## Status / completion

There is **no callback** to the API. The job writes its run status to MLflow
(`RUNNING → FINISHED/FAILED`); the playground reads MLflow. `gcloud run jobs
executions describe` is an ops-only secondary signal for container crashes.

## Verify (no cloud spend)

```bash
make validate     # terraform fmt -check + validate
make build-train  # docker build the training image
make dry-run      # run MODE=train against a local sqlite MLflow, assert exit 0 + a FINISHED run
```

## Show / destroy (deliberate, cost-approved)

```bash
# one-time: add the Neon DSN (must include ?sslmode=require)
printf '%s' "$NEON_DSN" | gcloud secrets versions add mini-networks-neon-dsn --data-file=-

make show         # terraform apply
gcloud pubsub topics publish mini-networks-train-requests \
  --message '{"model":"classifier","training_tier":"S","run_name":"smoke-1"}'
make destroy      # tear down; the train image stays in Artifact Registry
```

## Gotchas

- **GPU is region-gated.** The L4 job is `enable_gpu_job`-guarded (default off).
  Confirm europe-west1 Cloud Run GPU before enabling, else use an ephemeral
  GCE-L4 VM running the same image (documented fallback, not Terraform).
- **Trigger needs `roles/run.developer`** (not `run.invoker`) for `runJob` with
  overrides, plus `actAs` on the runtime SA.
- **Neon over public TLS** — DSN must include `?sslmode=require`; no VPC connector
  needed (and no Cloud SQL, per the workspace policy).
- **Secret value never in code** — only the secret *resource* is in Terraform.
