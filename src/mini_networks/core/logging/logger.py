"""Unified logger: writes metrics.jsonl + config.yaml + artifact paths."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


class Logger:
    def __init__(self, output_dir: str, run_name: str | None = None):
        ts = run_name or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(output_dir) / ts
        self.artifacts_dir = self.run_dir / "artifacts"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._config_path = self.run_dir / "config.yaml"

    def log_config(self, config_dict: dict) -> None:
        with open(self._config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def log_metric(self, step: int, key: str, value: Any) -> None:
        entry = {"step": step, "key": key, "value": value}
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_metrics(self, step: int, metrics: dict) -> None:
        for key, value in metrics.items():
            self.log_metric(step, key, value)

    def artifact_path(self, name: str) -> Path:
        return self.artifacts_dir / name

    def read_metrics(self) -> list[dict]:
        if not self._metrics_path.exists():
            return []
        lines = []
        with open(self._metrics_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
        return lines

    def latest_metrics(self, n: int = 10) -> list[dict]:
        all_metrics = self.read_metrics()
        return all_metrics[-n:]
