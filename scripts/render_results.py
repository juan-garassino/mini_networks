#!/usr/bin/env python
"""Inject sweep results into docs/ chapters between result markers.

Each chapter may contain one or more blocks:

    <!-- results:start items=classifier,resnet,vit -->
    ...replaced content...
    <!-- results:end -->

Run after a check sweep: `uv run python scripts/render_results.py`.
Idempotent — running twice produces no diff.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BLOCK = re.compile(
    r"(<!-- results:start items=(?P<items>[\w,\- ]+) -->)(?P<body>.*?)(<!-- results:end -->)",
    re.DOTALL,
)


def latest_report(sweep_root: Path) -> dict | None:
    reports = sorted(sweep_root.glob("*/report.json"))
    if not reports:
        return None
    return json.loads(reports[-1].read_text())


def results_table(report: dict, items: list[str]) -> str:
    by_name = {r["name"]: r for r in report["results"]}
    meta = report.get("meta", {})
    lines = [
        "",
        f"_Latest sweep: tier {meta.get('tier', '?')} on {meta.get('device', '?')}_",
        "",
        "| Item | Status | Metric | Value | Threshold |",
        "|---|---|---|---|---|",
    ]
    for name in items:
        r = by_name.get(name)
        if r is None:
            lines.append(f"| {name} | _not in last sweep_ | | | |")
            continue
        value = f"{r['value']:.4f}" if r.get("value") is not None else "n/a"
        threshold = f"{r['threshold']:.4f}" if r.get("threshold") is not None else "n/a"
        lines.append(f"| {name} | {r['status']} | {r.get('metric') or 'n/a'} | {value} | {threshold} |")
    lines.append("")
    return "\n".join(lines)


def render_file(path: Path, report: dict) -> bool:
    text = path.read_text()

    def repl(match: re.Match) -> str:
        items = [i.strip() for i in match.group("items").split(",") if i.strip()]
        return f"{match.group(1)}\n{results_table(report, items)}\n{match.group(4)}"

    new = BLOCK.sub(repl, text)
    if new != text:
        path.write_text(new)
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", default=str(ROOT / "runs" / "sweep"))
    parser.add_argument("--docs", default=str(ROOT / "docs"))
    args = parser.parse_args()

    report = latest_report(Path(args.sweep_root))
    if report is None:
        print(f"no report.json found under {args.sweep_root}", file=sys.stderr)
        return 1

    changed = []
    for path in sorted(Path(args.docs).glob("*.md")):
        if render_file(path, report):
            changed.append(path.name)
    print(f"updated: {', '.join(changed) if changed else 'nothing (already current)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
