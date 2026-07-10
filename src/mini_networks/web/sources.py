"""Run sources: read training runs from the local ``runs/`` tree or from MLflow,
normalized to the same schemas, plus a Composite that unions the in-memory job
store (for freshly-dispatched cloud jobs) with the persistent source.

Selected by env: ``MN_RUN_SOURCE`` = ``local`` (default) | ``mlflow``.
All MLflow imports are lazy so the base package imports without it.
"""
from __future__ import annotations

import logging
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mini_networks.api.schemas.web import (
    ConfigResponse,
    MetricSeries,
    MetricsResponse,
    RunSummary,
    SummaryResponse,
)
from mini_networks.web.metrics import pivot_long_to_series, read_jsonl, tail_latest

log = logging.getLogger(__name__)

_RUN_MARKERS = ("metrics.jsonl", "config.yaml", "summary.json", "training_state.pt")


class RunNotFound(KeyError):
    """Raised by a source when a run id cannot be resolved."""


class RunSource:
    """Interface implemented by every source. Methods raise RunNotFound on miss."""

    def list_runs(self) -> list[RunSummary]:  # pragma: no cover - interface
        raise NotImplementedError

    def get_metrics(self, run_id: str, since: int | None = None) -> MetricsResponse:  # pragma: no cover
        raise NotImplementedError

    def get_config(self, run_id: str) -> ConfigResponse:  # pragma: no cover
        raise NotImplementedError

    def get_summary(self, run_id: str) -> SummaryResponse:  # pragma: no cover
        raise NotImplementedError

    def open_artifact(self, run_id: str, name: str) -> tuple[Path, str]:  # pragma: no cover
        raise NotImplementedError


def _series_response(run_id: str, rows: list[dict], since: int | None) -> MetricsResponse:
    pivoted = pivot_long_to_series(rows, since)
    series = [MetricSeries(key=k, points=pts) for k, pts in pivoted]
    last = max((pts[-1][0] for _, pts in pivoted if pts), default=None)
    return MetricsResponse(run_id=run_id, series=series, latest_step=last)


# --------------------------------------------------------------------------- #
# Local runs/ directory
# --------------------------------------------------------------------------- #
def _looks_like_run(d: Path) -> bool:
    return d.is_dir() and any((d / m).exists() for m in _RUN_MARKERS)


class LocalRunsSource(RunSource):
    def __init__(self, runs_base: str | None = None):
        self.base = Path(runs_base or os.environ.get("MINI_NETWORKS_RUNS", "runs"))

    def _iter_run_dirs(self):
        if not self.base.exists():
            return
        for model_dir in sorted(self.base.iterdir()):
            if not model_dir.is_dir():
                continue
            for ts_dir in sorted(model_dir.iterdir()):
                if not ts_dir.is_dir():
                    continue
                if _looks_like_run(ts_dir):
                    yield ts_dir
                else:  # double-timestamp nesting: descend one level
                    for child in sorted(ts_dir.iterdir()):
                        if _looks_like_run(child):
                            yield child

    def _resolve(self, run_id: str) -> Path:
        d = self.base / run_id
        if _looks_like_run(d):
            return d
        raise RunNotFound(run_id)

    # A local run is only plausibly "live" if its metrics file was touched very
    # recently; many trainers never write summary.json, so a stale metrics file
    # without a summary just means the process finished and didn't summarize —
    # not that it's still running (which is what made old runs show a "REC" dot).
    _LIVE_WINDOW_S = 120

    def _status(self, d: Path) -> str:
        summary_path = d / "summary.json"
        if summary_path.exists():
            try:
                import json

                status = json.loads(summary_path.read_text()).get("status", "completed")
            except Exception:
                status = "completed"
            return "failed" if status == "failed" else "done"
        metrics = d / "metrics.jsonl"
        if metrics.exists():
            import time

            recent = (time.time() - metrics.stat().st_mtime) < self._LIVE_WINDOW_S
            return "running" if recent else "done"
        return "unknown"

    def _summary_for(self, d: Path) -> RunSummary:
        rid = str(d.relative_to(self.base))
        rows = read_jsonl(d / "metrics.jsonl")
        last_step, last_metrics = tail_latest(rows)
        artifacts_dir = d / "artifacts"
        names = sorted(p.name for p in artifacts_dir.iterdir()) if artifacts_dir.exists() else []
        created = datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc).isoformat()
        return RunSummary(
            id=rid,
            model=Path(rid).parts[0],
            source="local",
            status=self._status(d),
            run_name=d.name,
            created_at=created,
            last_step=last_step,
            last_metrics=last_metrics,
            artifact_names=names,
        )

    def list_runs(self) -> list[RunSummary]:
        return [self._summary_for(d) for d in self._iter_run_dirs()]

    def get_metrics(self, run_id: str, since: int | None = None) -> MetricsResponse:
        d = self._resolve(run_id)
        return _series_response(run_id, read_jsonl(d / "metrics.jsonl"), since)

    def get_config(self, run_id: str) -> ConfigResponse:
        d = self._resolve(run_id)
        cfg_path = d / "config.yaml"
        config = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        return ConfigResponse(run_id=run_id, config=config or {})

    def get_summary(self, run_id: str) -> SummaryResponse:
        d = self._resolve(run_id)
        import json

        s_path = d / "summary.json"
        summary = json.loads(s_path.read_text()) if s_path.exists() else {}
        return SummaryResponse(run_id=run_id, summary=summary or {})

    def open_artifact(self, run_id: str, name: str) -> tuple[Path, str]:
        d = self._resolve(run_id)
        path = (d / "artifacts" / name).resolve()
        artifacts_root = (d / "artifacts").resolve()
        if artifacts_root not in path.parents or not path.exists():
            raise RunNotFound(f"{run_id}/{name}")
        media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return path, media


# --------------------------------------------------------------------------- #
# MLflow (Neon + GCS), read server-lessly via MlflowClient on the DB URI
# --------------------------------------------------------------------------- #
_MLFLOW_STATUS = {
    "RUNNING": "running",
    "FINISHED": "done",
    "FAILED": "failed",
    "KILLED": "failed",
    "SCHEDULED": "pending",
}


class MLflowSource(RunSource):
    def __init__(self, tracking_uri: str | None = None, experiment: str | None = None):
        from mlflow.tracking import MlflowClient

        self._client = MlflowClient(tracking_uri=tracking_uri or os.environ.get("MN_MLFLOW_TRACKING_URI"))
        self._experiment = experiment or os.environ.get("MN_MLFLOW_EXPERIMENT", "mini-networks")

    def _exp_id(self) -> str | None:
        exp = self._client.get_experiment_by_name(self._experiment)
        return exp.experiment_id if exp else None

    def _get_run(self, run_id: str):
        try:
            return self._client.get_run(run_id)
        except Exception:
            raise RunNotFound(run_id)

    def _summary_for(self, run) -> RunSummary:
        tags = run.data.tags or {}
        run_dir = tags.get("run_dir", "")
        model = Path(run_dir).parent.name if run_dir else (run.info.run_name or "").split("-")[0]
        created = (
            datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc).isoformat()
            if run.info.start_time
            else None
        )
        return RunSummary(
            id=run.info.run_id,
            model=model or "unknown",
            source="mlflow",
            status=_MLFLOW_STATUS.get(run.info.status, "unknown"),
            run_name=run.info.run_name,
            created_at=created,
            last_step=None,
            last_metrics={k: float(v) for k, v in (run.data.metrics or {}).items()},
            artifact_names=[a.path for a in self._client.list_artifacts(run.info.run_id)],
        )

    def list_runs(self) -> list[RunSummary]:
        exp_id = self._exp_id()
        if exp_id is None:
            return []
        return [self._summary_for(r) for r in self._client.search_runs([exp_id])]

    def get_metrics(self, run_id: str, since: int | None = None) -> MetricsResponse:
        run = self._get_run(run_id)
        rows: list[dict] = []
        for key in (run.data.metrics or {}):
            for m in self._client.get_metric_history(run_id, key):
                rows.append({"step": m.step, "key": key, "value": m.value})
        return _series_response(run_id, rows, since)

    def get_config(self, run_id: str) -> ConfigResponse:
        run = self._get_run(run_id)
        return ConfigResponse(run_id=run_id, config=dict(run.data.params or {}))

    def get_summary(self, run_id: str) -> SummaryResponse:
        run = self._get_run(run_id)
        tags = {k: v for k, v in (run.data.tags or {}).items() if not k.startswith("mlflow.")}
        return SummaryResponse(run_id=run_id, summary=tags)

    def open_artifact(self, run_id: str, name: str) -> tuple[Path, str]:
        self._get_run(run_id)
        try:
            local = self._client.download_artifacts(run_id, name)
        except Exception:
            raise RunNotFound(f"{run_id}/{name}")
        media = mimetypes.guess_type(name)[0] or "application/octet-stream"
        return Path(local), media


# --------------------------------------------------------------------------- #
# Composite: persistent source ∪ in-memory job store (cloud dispatched stubs)
# --------------------------------------------------------------------------- #
_JOBSTATUS = {"pending": "pending", "running": "running", "done": "done",
              "failed": "failed", "dispatched": "dispatched"}


class CompositeSource(RunSource):
    def __init__(self, persistent: RunSource):
        self._persistent = persistent

    def list_runs(self) -> list[RunSummary]:
        from mini_networks.api.dependencies import list_jobs

        runs = self._persistent.list_runs()
        known = {r.run_name for r in runs if r.run_name} | {r.id for r in runs}
        for job in list_jobs():
            if job.job_id in known:
                continue  # persistent store already has the authoritative status
            tail = {m["key"]: float(m["value"]) for m in job.metrics_tail
                    if isinstance(m.get("value"), (int, float))}
            runs.append(RunSummary(
                id=job.job_id, model=job.model, source="jobstore",
                status=_JOBSTATUS.get(job.status, "unknown"), run_name=job.job_id,
                last_metrics=tail,
            ))
        return runs

    def get_metrics(self, run_id: str, since: int | None = None) -> MetricsResponse:
        try:
            return self._persistent.get_metrics(run_id, since)
        except RunNotFound:
            self._require_job(run_id)
            return MetricsResponse(run_id=run_id, series=[], latest_step=None)

    def get_config(self, run_id: str) -> ConfigResponse:
        try:
            return self._persistent.get_config(run_id)
        except RunNotFound:
            self._require_job(run_id)
            return ConfigResponse(run_id=run_id, config={})

    def get_summary(self, run_id: str) -> SummaryResponse:
        try:
            return self._persistent.get_summary(run_id)
        except RunNotFound:
            job = self._require_job(run_id)
            return SummaryResponse(run_id=run_id, summary={"status": job.status})

    def open_artifact(self, run_id: str, name: str) -> tuple[Path, str]:
        return self._persistent.open_artifact(run_id, name)

    def _require_job(self, run_id: str):
        from mini_networks.api.dependencies import get_job

        job = get_job(run_id)
        if job is None:
            raise RunNotFound(run_id)
        return job


def get_run_source() -> RunSource:
    kind = os.environ.get("MN_RUN_SOURCE", "local")
    persistent: RunSource = MLflowSource() if kind == "mlflow" else LocalRunsSource()
    return CompositeSource(persistent)
