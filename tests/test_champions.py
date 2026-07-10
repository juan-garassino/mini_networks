"""Champion loader: pull/resolve logic with a hand-rolled stub MlflowClient."""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from mini_networks.cloud import champions


class _StubClient:
    def __init__(self, versions_by_model=None, fail_download=False):
        self._versions = versions_by_model or {}
        self._fail_download = fail_download
        self.calls = []

    def get_latest_versions(self, name, stages=None):
        if name not in self._versions:
            raise RuntimeError(f"RESOURCE_DOES_NOT_EXIST: {name} not found")
        return self._versions[name]

    def download_artifacts(self, run_id, path, dst_path=None):
        if self._fail_download:
            raise RuntimeError("gcs down")
        if dst_path is not None and not Path(dst_path).exists():
            # real mlflow raises when the destination doesn't pre-exist
            raise RuntimeError("The destination path for downloaded artifacts does not exist!")
        self.calls.append(("download", run_id, path))
        d = Path(dst_path) / path
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.pt").write_bytes(b"ckpt")
        return str(d)


def _mv(version="3", run_id="run-9"):
    return types.SimpleNamespace(version=version, run_id=run_id)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("MN_MLFLOW_TRACKING_URI", "https://tracker.example")


def test_disabled_without_tracking_uri(monkeypatch, tmp_path):
    monkeypatch.delenv("MN_MLFLOW_TRACKING_URI", raising=False)
    status = champions.pull_champions(["classifier"], checkpoint_root=tmp_path)
    assert "MN_MLFLOW_TRACKING_URI unset" in status["classifier"]


def test_pull_places_ckpt_in_artifacts_layout(enabled, monkeypatch, tmp_path):
    client = _StubClient({"mini-classifier": [_mv()]})
    monkeypatch.setattr(champions, "_make_client", lambda: client)
    status = champions.pull_champions(["classifier"], checkpoint_root=tmp_path)
    assert status == {"classifier": "v3"}
    dest = champions.champion_artifacts_dir("classifier", tmp_path)
    assert (dest / "model.pt").read_bytes() == b"ckpt"
    assert champions.has_champion("classifier", tmp_path)
    assert "mini-classifier v3 run run-9" in (dest.parent / "VERSION").read_text()


def test_missing_registered_model_is_no_champion(enabled, monkeypatch, tmp_path):
    monkeypatch.setattr(champions, "_make_client", lambda: _StubClient({}))
    status = champions.pull_champions(["classifier"], checkpoint_root=tmp_path)
    assert status == {"classifier": "no champion"}
    assert not champions.has_champion("classifier", tmp_path)


def test_empty_production_stage_is_no_champion(enabled, monkeypatch, tmp_path):
    monkeypatch.setattr(champions, "_make_client", lambda: _StubClient({"mini-gan": []}))
    assert champions.pull_champions(["gan"], checkpoint_root=tmp_path) == {"gan": "no champion"}


def test_download_error_never_raises(enabled, monkeypatch, tmp_path):
    client = _StubClient({"mini-classifier": [_mv()]}, fail_download=True)
    monkeypatch.setattr(champions, "_make_client", lambda: client)
    status = champions.pull_champions(["classifier"], checkpoint_root=tmp_path)
    assert status["classifier"].startswith("error:")


def test_has_champion_requires_pt_files(tmp_path):
    d = champions.champion_artifacts_dir("vae", tmp_path)
    d.mkdir(parents=True)
    assert not champions.has_champion("vae", tmp_path)
    (d / "model.pt").write_bytes(b"x")
    assert champions.has_champion("vae", tmp_path)
