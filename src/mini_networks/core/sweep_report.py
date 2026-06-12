"""Sweep check-report data model and writers.

Pure stdlib so it stays unit-testable without torch. The gate
(colab/gate.py) produces CheckResult records; this module renders them to
runs/sweep/<timestamp>/report.{md,json}.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CheckResult:
    item_type: str                      # "model" | "composition"
    name: str
    status: str                         # "pass" | "fail" | "error"
    tier: str
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None
    higher_is_better: bool = True
    s_check: dict = field(default_factory=dict)   # {"finite": bool, "trend": str, "keys": [...]}
    infer_summary: str | None = None
    roundtrip: str = "skipped"          # "ok" | "skipped" | "failed: ..."
    duration_s: float = 0.0
    run_dir: str | None = None
    error: str | None = None            # traceback tail when status == "error"


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def render_markdown(results: list[CheckResult], meta: dict) -> str:
    lines = ["# Sweep Check Report", ""]
    for key in sorted(meta):
        lines.append(f"- **{key}**: {meta[key]}")
    counts = {s: sum(1 for r in results if r.status == s) for s in ("pass", "fail", "error")}
    lines += [
        "",
        f"**{counts['pass']} pass / {counts['fail']} fail / {counts['error']} error** "
        f"({len(results)} items)",
        "",
        "| Name | Type | Status | Metric | Value | Threshold | Round-trip | Duration |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.item_type} | {r.status} | {r.metric or 'n/a'} "
            f"| {_fmt(r.value)} | {_fmt(r.threshold)} | {r.roundtrip} | {r.duration_s:.1f}s |"
        )
    failures = [r for r in results if r.status != "pass"]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines.append(f"### {r.name} ({r.status})")
            if r.s_check:
                lines.append(f"- s_check: `{r.s_check}`")
            if r.error:
                lines += ["", "```", r.error.strip(), "```", ""]
    return "\n".join(lines) + "\n"


def write_report(results: list[CheckResult], sweep_dir: str | Path, meta: dict) -> tuple[Path, Path]:
    """Write report.md + report.json into sweep_dir; returns both paths."""
    out = Path(sweep_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report.json"
    md_path = out / "report.md"
    payload = {"meta": meta, "results": [asdict(r) for r in results]}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    md_path.write_text(render_markdown(results, meta))
    return md_path, json_path
