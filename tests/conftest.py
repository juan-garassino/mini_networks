"""Shared test helpers.

Some datasets (FSDD speech digits, Iris) are fetched from the network on first
use; tests run with require_downloads=False so an absent local cache raises
RuntimeError("... downloads disabled"). Those tests must skip, not fail —
CI has no dataset cache either.
"""
from __future__ import annotations

import contextlib
import json

import pytest
import yaml


@contextlib.contextmanager
def dataset_or_skip():
    try:
        yield
    except RuntimeError as exc:
        if "downloads disabled" in str(exc):
            pytest.skip(str(exc))
        raise


@pytest.fixture
def skip_if_dataset_missing():
    return dataset_or_skip


def _write_run(run_dir, *, metrics, config=None, summary=None, state=False, artifacts=()):
    """Synthesize a run directory matching the Logger on-disk contract."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    with open(run_dir / "metrics.jsonl", "w") as f:
        for step, key, value in metrics:
            f.write(json.dumps({"step": step, "key": key, "value": value}) + "\n")
    if config is not None:
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
    if summary is not None:
        (run_dir / "summary.json").write_text(json.dumps(summary))
    if state:
        (run_dir / "training_state.pt").write_bytes(b"state")
    for name, data in artifacts:
        (run_dir / "artifacts" / name).write_bytes(data)


@pytest.fixture
def runs_dir(tmp_path):
    """A runs/ tree exercising done / live / nested-double-timestamp shapes."""
    base = tmp_path / "runs"
    # Completed run (summary present).
    _write_run(
        base / "vae" / "vae-001",
        metrics=[(0, "loss", 1.5), (0, "epoch", 0), (1, "loss", 0.8), (1, "epoch", 1)],
        config={"lr": 0.001, "batch_size": 64},
        summary={"status": "completed", "epochs": 2},
        artifacts=[("model.pt", b"weights"), ("sample.png", b"\x89PNG\r\n")],
    )
    # Live run (state but no summary).
    _write_run(
        base / "gan" / "gan-001",
        metrics=[(0, "loss", 2.0)],
        config={"lr": 0.0002},
        state=True,
    )
    # Double-timestamp nesting (the historical quirk).
    _write_run(
        base / "clip" / "212716" / "212716",
        metrics=[(0, "loss", 3.1)],
        config={"embed_dim": 128},
        summary={"status": "completed", "epochs": 1},
    )
    return base
