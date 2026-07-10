"""Pure metric helpers — no torch, no mlflow, trivially unit-testable.

The on-disk metrics format is LONG: one JSON object per line,
``{"step": int, "key": str, "value": Any}``, with multiple keys per step.
The frontend wants per-key series, so the core operation is a pivot.
"""
from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def pivot_long_to_series(rows: list[dict], since: int | None = None) -> list[tuple[str, list[tuple[int, float]]]]:
    """Group LONG rows by key into sorted (step, value) series.

    Drops non-numeric values; ``since`` keeps only steps strictly greater than it
    (incremental polling for live runs).
    """
    grouped: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        step, key, value = r.get("step"), r.get("key"), r.get("value")
        if key is None or not _is_number(step) or not _is_number(value):
            continue
        if since is not None and int(step) <= since:
            continue
        grouped.setdefault(key, []).append((int(step), float(value)))
    return [(key, sorted(grouped[key])) for key in sorted(grouped)]


def latest_step(rows: list[dict]) -> int | None:
    steps = [int(r["step"]) for r in rows if _is_number(r.get("step"))]
    return max(steps) if steps else None


def tail_latest(rows: list[dict]) -> tuple[int | None, dict[str, float]]:
    """The last step and a {key: value} dict of that step's numeric metrics."""
    ls = latest_step(rows)
    if ls is None:
        return None, {}
    out: dict[str, float] = {}
    for r in rows:
        if _is_number(r.get("step")) and int(r["step"]) == ls and _is_number(r.get("value")):
            out[r["key"]] = float(r["value"])
    return ls, out
