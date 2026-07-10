"""Champion/challenger registration into the global MLflow Model Registry.

Mirrors the desktop garassino-ml ``mlflow_tracking.log_and_promote`` contract:
a gate-passing M/L training registers its checkpoint(s) as a new version of
registered model ``mini-<name>``; the version whose gate metric beats the
current Production champion is promoted (archiving the old champion), else it
lands in Staging. The comparison reads the ``gate_value`` tag stored on the
champion version, so promotion is self-contained — it never depends on run
metric keys. Every call is wrapped: a tracker hiccup can never fail a run.
Enabled by ``MN_MLFLOW_REGISTER=1`` on top of ``MN_MLFLOW_TRACKING_URI``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from mini_networks.core.logging.mlflow_sink import TRACKING_URI_ENV

log = logging.getLogger(__name__)

REGISTER_ENV = "MN_MLFLOW_REGISTER"
MODEL_PREFIX = "mini-"


def is_register_enabled() -> bool:
    return os.environ.get(REGISTER_ENV) == "1" and bool(os.environ.get(TRACKING_URI_ENV))


def _make_client():
    # Function-local import + tiny factory so tests can stub the client and the
    # base install never imports mlflow.
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=os.environ[TRACKING_URI_ENV])


def register_and_promote(
    name: str,
    artifacts_dir: str | Path,
    metric_key: str | None,
    value: float | None,
    higher_is_better: bool,
    run_id: str | None,
    tier: str,
    min_delta: float = 0.0,
) -> dict:
    """Register the run's checkpoint(s) as ``mini-<name>`` and promote if it wins.

    Returns {"tracked": bool, ...}; tracked=False (with a reason) when disabled,
    when the run has no MLflow id / metric / checkpoint, or on any tracker error.
    """
    if not is_register_enabled():
        return {"tracked": False, "reason": "disabled"}
    if run_id is None:
        return {"tracked": False, "reason": "no mlflow run"}
    if metric_key is None or value is None:
        return {"tracked": False, "reason": "no gate metric"}
    ckpts = sorted(Path(artifacts_dir).glob("*.pt"))
    if not ckpts:
        return {"tracked": False, "reason": "no checkpoint under artifacts/"}
    try:
        client = _make_client()
        model_name = f"{MODEL_PREFIX}{name}"
        for ckpt in ckpts:
            client.log_artifact(run_id, str(ckpt), artifact_path="model")
        try:
            client.create_registered_model(model_name)
        except Exception:
            pass  # already exists
        source = f"{client.get_run(run_id).info.artifact_uri}/model"
        mv = client.create_model_version(model_name, source=source, run_id=run_id)

        promote = True
        prod = client.get_latest_versions(model_name, stages=["Production"])
        if prod:
            champ = (prod[0].tags or {}).get("gate_value")
            if champ is not None:
                promote = (
                    value > float(champ) + min_delta
                    if higher_is_better
                    else value < float(champ) - min_delta
                )

        client.set_model_version_tag(model_name, mv.version, "gate_metric", metric_key)
        client.set_model_version_tag(model_name, mv.version, "gate_value", repr(float(value)))
        client.set_model_version_tag(model_name, mv.version, "tier", tier)
        stage = "Production" if promote else "Staging"
        client.transition_model_version_stage(
            model_name, mv.version, stage=stage, archive_existing_versions=promote
        )
        log.info("registry: %s v%s %s=%.4f -> %s", model_name, mv.version, metric_key, value, stage)
        return {
            "tracked": True,
            "model": model_name,
            "version": str(mv.version),
            "stage": stage,
            "promoted": promote,
        }
    except Exception as e:  # never fail the gate on a tracker glitch
        log.warning("registry: skipped (%s: %s)", type(e).__name__, e)
        return {"tracked": False, "reason": f"{type(e).__name__}: {e}"}
