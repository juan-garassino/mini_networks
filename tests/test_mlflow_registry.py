"""Champion/challenger registry logic against a hand-rolled stub MlflowClient.

Covers the env gate, the missing-run/metric/checkpoint guards, first-version
promotion, challenger-vs-champion in both metric directions, and the
never-raise wrapper (a tracker error returns tracked=False, no exception).
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from mini_networks.core.logging import mlflow_registry


class _StubModelVersion:
    def __init__(self, version, tags=None):
        self.version = version
        self.tags = tags or {}


class _StubClient:
    def __init__(self, champion_value=None):
        self.calls = []
        self._champion = champion_value
        self._next_version = 1

    def log_artifact(self, run_id, path, artifact_path=None):
        self.calls.append(("log_artifact", run_id, Path(path).name, artifact_path))

    def create_registered_model(self, name):
        self.calls.append(("create_registered_model", name))

    def get_run(self, run_id):
        info = types.SimpleNamespace(artifact_uri=f"mlflow-artifacts:/1/{run_id}/artifacts")
        return types.SimpleNamespace(info=info)

    def create_model_version(self, name, source, run_id):
        self.calls.append(("create_model_version", name, source, run_id))
        version = _StubModelVersion(self._next_version)
        self._next_version += 1
        return version

    def get_latest_versions(self, name, stages=None):
        if self._champion is None:
            return []
        return [_StubModelVersion(1, tags={"gate_value": repr(self._champion)})]

    def set_model_version_tag(self, name, version, key, value):
        self.calls.append(("set_model_version_tag", name, version, key, value))

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions):
        self.calls.append(("transition", name, version, stage, archive_existing_versions))


class _ExplodingClient(_StubClient):
    def create_model_version(self, name, source, run_id):
        raise RuntimeError("registry down")


@pytest.fixture
def artifacts_dir(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"ckpt")
    return tmp_path


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("MN_MLFLOW_TRACKING_URI", "https://tracker.example")
    monkeypatch.setenv("MN_MLFLOW_REGISTER", "1")


def _call(artifacts_dir, value=0.9, higher_is_better=True, run_id="run-1", metric="accuracy"):
    return mlflow_registry.register_and_promote(
        name="classifier",
        artifacts_dir=artifacts_dir,
        metric_key=metric,
        value=value,
        higher_is_better=higher_is_better,
        run_id=run_id,
        tier="M",
    )


def test_disabled_without_register_env(monkeypatch, artifacts_dir):
    monkeypatch.setenv("MN_MLFLOW_TRACKING_URI", "https://tracker.example")
    monkeypatch.delenv("MN_MLFLOW_REGISTER", raising=False)
    assert _call(artifacts_dir) == {"tracked": False, "reason": "disabled"}


def test_disabled_without_tracking_uri(monkeypatch, artifacts_dir):
    monkeypatch.delenv("MN_MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MN_MLFLOW_REGISTER", "1")
    assert _call(artifacts_dir) == {"tracked": False, "reason": "disabled"}


def test_guards(enabled, monkeypatch, artifacts_dir, tmp_path):
    monkeypatch.setattr(mlflow_registry, "_make_client", lambda: _StubClient())
    assert _call(artifacts_dir, run_id=None)["reason"] == "no mlflow run"
    assert _call(artifacts_dir, value=None)["reason"] == "no gate metric"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "no checkpoint" in _call(empty)["reason"]


def test_first_version_promotes(enabled, monkeypatch, artifacts_dir):
    client = _StubClient(champion_value=None)
    monkeypatch.setattr(mlflow_registry, "_make_client", lambda: client)
    out = _call(artifacts_dir, value=0.8)
    assert out["tracked"] and out["stage"] == "Production" and out["promoted"]
    assert ("log_artifact", "run-1", "model.pt", "model") in client.calls
    assert ("transition", "mini-classifier", 1, "Production", True) in client.calls


def test_challenger_loses_goes_to_staging(enabled, monkeypatch, artifacts_dir):
    client = _StubClient(champion_value=0.95)
    monkeypatch.setattr(mlflow_registry, "_make_client", lambda: client)
    out = _call(artifacts_dir, value=0.8)
    assert out["stage"] == "Staging" and not out["promoted"]
    assert ("transition", "mini-classifier", 1, "Staging", False) in client.calls


def test_challenger_wins_archives_champion(enabled, monkeypatch, artifacts_dir):
    client = _StubClient(champion_value=0.7)
    monkeypatch.setattr(mlflow_registry, "_make_client", lambda: client)
    out = _call(artifacts_dir, value=0.8)
    assert out["stage"] == "Production" and out["promoted"]


def test_lower_is_better_direction(enabled, monkeypatch, artifacts_dir):
    client = _StubClient(champion_value=1.5)
    monkeypatch.setattr(mlflow_registry, "_make_client", lambda: client)
    out = _call(artifacts_dir, value=1.2, higher_is_better=False, metric="eval_loss")
    assert out["stage"] == "Production"
    out = _call(artifacts_dir, value=2.0, higher_is_better=False, metric="eval_loss")
    assert out["stage"] == "Staging"


def test_tracker_error_never_raises(enabled, monkeypatch, artifacts_dir):
    monkeypatch.setattr(mlflow_registry, "_make_client", lambda: _ExplodingClient())
    out = _call(artifacts_dir)
    assert out["tracked"] is False and "registry down" in out["reason"]
