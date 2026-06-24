"""Pub/Sub-triggered Cloud Function: launch an ephemeral Cloud Run training job.

A message ``{"model","training_tier","epochs","device","run_name","hparams"}``
launches a Cloud Run Job execution with per-run env overrides. Fire-and-forget:
the job writes its status to MLflow; the frontend reads MLflow. We never wait.

IAM: the function's service account needs ``roles/run.developer`` (for
``runJob`` with overrides — ``run.invoker`` is NOT enough) plus ``actAs`` on the
job's runtime service account.
"""
from __future__ import annotations

import base64
import json
import logging
import os

import functions_framework
from google.cloud import run_v2

log = logging.getLogger(__name__)

PROJECT = os.environ["GCP_PROJECT"]
REGION = os.environ["GCP_REGION"]
CPU_JOB = os.environ["CPU_JOB_NAME"]
GPU_JOB = os.environ.get("GPU_JOB_NAME", "")
GPU_MODELS = {m for m in os.environ.get("GPU_MODELS", "").split(",") if m}


def _env(name: str, value: str) -> run_v2.EnvVar:
    return run_v2.EnvVar(name=name, value=value)


@functions_framework.cloud_event
def on_train_request(event):
    msg = json.loads(base64.b64decode(event.data["message"]["data"]))
    model = msg["model"]
    tier = msg.get("training_tier", "M")
    run_name = msg.get("run_name") or f"{model}-run"

    job = GPU_JOB if (model in GPU_MODELS and GPU_JOB) else CPU_JOB
    name = f"projects/{PROJECT}/locations/{REGION}/jobs/{job}"

    overrides = run_v2.RunJobRequest.Overrides(
        container_overrides=[
            run_v2.RunJobRequest.Overrides.ContainerOverride(
                env=[
                    _env("MODEL", model),
                    _env("TRAINING_TIER", tier),
                    _env("RUN_NAME", run_name),
                    _env("DEVICE", msg.get("device", "cpu")),
                    _env("EPOCHS", str(msg.get("epochs", ""))),
                    _env("HPARAMS", json.dumps(msg.get("hparams", {}))),
                ]
            )
        ]
    )
    run_v2.JobsClient().run_job(run_v2.RunJobRequest(name=name, overrides=overrides))
    log.info("Launched %s for model=%s run=%s", job, model, run_name)
