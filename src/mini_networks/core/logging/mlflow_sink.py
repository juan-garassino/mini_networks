"""Optional MLflow sink for Logger.

Import-safe when mlflow is absent: the module imports with only stdlib, and the
heavy ``import mlflow`` happens lazily inside ``MLflowSink.__init__``. Every call
is wrapped so a flaky tracking backend can never break a training run — the file
logging in ``Logger`` is the source of truth; MLflow is an additive mirror.

Activated by ``MN_MLFLOW_TRACKING_URI`` (a database URI like
``postgresql://…`` or ``sqlite:///…`` works server-lessly). Artifacts go to
``MN_MLFLOW_ARTIFACT_ROOT`` (e.g. a ``gs://`` path); experiment name from
``MN_MLFLOW_EXPERIMENT`` (default ``mini-networks``).
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

TRACKING_URI_ENV = "MN_MLFLOW_TRACKING_URI"
ARTIFACT_ROOT_ENV = "MN_MLFLOW_ARTIFACT_ROOT"
EXPERIMENT_ENV = "MN_MLFLOW_EXPERIMENT"
DEFAULT_EXPERIMENT = "mini-networks"

_PARAM_VALUE_MAX = 250  # conservative MLflow param-value length cap


def is_mlflow_enabled() -> bool:
    return bool(os.environ.get(TRACKING_URI_ENV))


def _flatten(d: dict, prefix: str = "", out: dict | None = None) -> dict:
    out = {} if out is None else out
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            _flatten(v, prefix=f"{key}.", out=out)
        else:
            out[key] = v
    return out


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class MLflowSink:
    """Mirrors Logger calls to an MLflow run via MlflowClient (no global run state)."""

    def __init__(self, run_name: str, run_dir: Path, artifacts_dir: Path, tags: dict | None = None):
        self._active = False
        self._ended = False
        self._client = None
        self._run_id = None
        self._artifacts_dir = Path(artifacts_dir)
        try:
            from mlflow.tracking import MlflowClient
        except Exception as e:  # pragma: no cover - exercised only when mlflow absent
            log.warning("MLflow requested but import failed; logging file-only: %s", e)
            return
        try:
            client = MlflowClient(tracking_uri=os.environ[TRACKING_URI_ENV])
            exp_id = self._ensure_experiment(client)
            run = client.create_run(
                exp_id,
                run_name=run_name,
                tags={"run_dir": str(run_dir), **(tags or {})},
            )
            self._client = client
            self._run_id = run.info.run_id
            self._active = True
        except Exception as e:
            log.warning("MLflow init failed; logging file-only: %s", e)

    def _ensure_experiment(self, client) -> str:
        name = os.environ.get(EXPERIMENT_ENV, DEFAULT_EXPERIMENT)
        exp = client.get_experiment_by_name(name)
        if exp is not None:
            return exp.experiment_id
        return client.create_experiment(name, artifact_location=os.environ.get(ARTIFACT_ROOT_ENV) or None)

    def log_params(self, config_dict: dict) -> None:
        if not self._active:
            return
        try:
            for k, v in _flatten(config_dict).items():
                self._client.log_param(self._run_id, k, str(v)[:_PARAM_VALUE_MAX])
        except Exception as e:
            log.warning("MLflow log_params failed: %s", e)

    def log_metric(self, step: int, key: str, value: Any) -> None:
        if not self._active or not _is_number(value):
            return
        try:
            self._client.log_metric(self._run_id, key, float(value), step=int(step))
        except Exception as e:
            log.warning("MLflow log_metric failed: %s", e)

    def set_tags(self, tags: dict) -> None:
        if not self._active:
            return
        try:
            for k, v in tags.items():
                self._client.set_tag(self._run_id, k, str(v)[:_PARAM_VALUE_MAX])
        except Exception as e:
            log.warning("MLflow set_tags failed: %s", e)

    def log_final_metrics(self, metrics: dict) -> None:
        if not self._active:
            return
        for k, v in metrics.items():
            self.log_metric(0, f"final_{k}", v)

    def _flush_artifacts(self) -> None:
        if not self._active:
            return
        try:
            d = self._artifacts_dir
            if d.exists() and any(d.iterdir()):
                self._client.log_artifacts(self._run_id, str(d))
        except Exception as e:
            log.warning("MLflow log_artifacts failed: %s", e)

    def close(self, status: str = "FINISHED") -> None:
        if not self._active or self._ended:
            return
        self._ended = True
        try:
            self._flush_artifacts()
            self._client.set_terminated(self._run_id, status=status)
        except Exception as e:
            log.warning("MLflow set_terminated failed: %s", e)
