"""Shared dependencies: job store and model registry."""
from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from mini_networks.api.schemas.training import JobStatus


# In-memory job store
_jobs: dict[str, JobStatus] = {}
_lock = threading.Lock()


def register_job(job_id: str, model: str, output_dir: str) -> None:
    with _lock:
        _jobs[job_id] = JobStatus(
            job_id=job_id,
            model=model,
            status="pending",
            output_dir=output_dir,
        )


def update_job(job_id: str, **kwargs) -> None:
    with _lock:
        if job_id in _jobs:
            job = _jobs[job_id]
            for k, v in kwargs.items():
                setattr(job, k, v)


def get_job(job_id: str) -> JobStatus | None:
    return _jobs.get(job_id)


def list_jobs() -> list[JobStatus]:
    return list(_jobs.values())


def make_job_id(model: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{model}-{ts}"


def make_output_dir(base: str, model: str, job_id: str) -> str:
    path = Path(base) / model / job_id
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


# Model registry moved to core (api re-exports it for back-compat)
from mini_networks.core.registry import get_model_registry  # noqa: E402,F401
