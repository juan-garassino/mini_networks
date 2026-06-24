"""MLflow sink: Logger mirrors to MLflow when enabled; file contract unchanged."""
from __future__ import annotations

import json

import pytest

mlflow = pytest.importorskip("mlflow")
from mlflow.tracking import MlflowClient  # noqa: E402

from mini_networks.core.logging.logger import Logger  # noqa: E402


def _enable(monkeypatch, tmp_path):
    monkeypatch.setenv("MN_MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("MN_MLFLOW_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MN_MLFLOW_EXPERIMENT", "test-exp")


def _client(tmp_path):
    return MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")


def test_logger_dual_writes_to_mlflow(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    logger = Logger(output_dir=str(tmp_path / "runs" / "vae"), run_name="vae-001")
    logger.log_config({"lr": 0.001, "batch_size": 64, "nested": {"a": 1}})
    logger.log_metrics(0, {"loss": 1.5, "accuracy": 0.2})
    logger.log_metrics(1, {"loss": 0.9, "accuracy": 0.5})
    logger.artifact_path("model.pt").write_bytes(b"weights")
    logger.log_summary({"status": "completed", "epochs": 2})
    logger.close()

    # File contract unchanged.
    rows = [json.loads(x) for x in (logger.run_dir / "metrics.jsonl").read_text().splitlines() if x.strip()]
    assert {"step": 0, "key": "loss", "value": 1.5} in rows
    assert (logger.run_dir / "config.yaml").exists()

    # MLflow mirror.
    client = _client(tmp_path)
    exp = client.get_experiment_by_name("test-exp")
    assert exp is not None
    found = client.search_runs([exp.experiment_id])
    assert len(found) == 1
    run = found[0]
    assert run.info.status == "FINISHED"
    assert run.data.params["lr"] == "0.001"
    assert run.data.params["nested.a"] == "1"
    hist = sorted(client.get_metric_history(run.info.run_id, "loss"), key=lambda h: h.step)
    assert [(h.step, h.value) for h in hist] == [(0, 1.5), (1, 0.9)]
    assert "model.pt" in [a.path for a in client.list_artifacts(run.info.run_id)]


def test_logger_no_mlflow_when_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("MN_MLFLOW_TRACKING_URI", raising=False)
    logger = Logger(output_dir=str(tmp_path / "runs" / "vae"), run_name="vae-x")
    assert logger._mlflow is None
    logger.log_metrics(0, {"loss": 1.0})
    logger.close()  # idempotent no-op, must not raise
    assert (logger.run_dir / "metrics.jsonl").exists()


def test_logger_failed_status(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    logger = Logger(output_dir=str(tmp_path / "runs" / "gan"), run_name="gan-001")
    logger.log_metrics(0, {"loss": 2.0})
    logger.close(status="FAILED")
    client = _client(tmp_path)
    exp = client.get_experiment_by_name("test-exp")
    assert client.search_runs([exp.experiment_id])[0].info.status == "FAILED"
