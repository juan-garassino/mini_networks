"""Checkpoint helpers for timestamped runs and automatic resume."""
from __future__ import annotations

from pathlib import Path


def latest_run_dir(base_dir: str | Path) -> Path | None:
    root = Path(base_dir)
    if not root.exists():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def find_resumable_run(base_dir: str | Path) -> Path | None:
    run_dir = latest_run_dir(base_dir)
    if run_dir is None:
        return None
    state_path = run_dir / "training_state.pt"
    if not state_path.exists():
        return None
    return run_dir
