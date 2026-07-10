"""Read-layer HTTP contract via TestClient over a fixture runs/ tree."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(runs_dir, monkeypatch):
    monkeypatch.setenv("MINI_NETWORKS_RUNS", str(runs_dir))
    monkeypatch.setenv("MN_RUN_SOURCE", "local")
    monkeypatch.delenv("MN_MLFLOW_TRACKING_URI", raising=False)
    from mini_networks.api.main import create_app

    return TestClient(create_app())


def test_list_runs(client):
    body = client.get("/web/runs").json()
    ids = {r["id"] for r in body["runs"]}
    assert {"vae/vae-001", "gan/gan-001", "clip/212716/212716"} <= ids


def test_metrics_endpoint(client):
    body = client.get("/web/runs/vae/vae-001/metrics").json()
    series = {s["key"]: s["points"] for s in body["series"]}
    assert series["loss"] == [[0, 1.5], [1, 0.8]]


def test_metrics_since(client):
    body = client.get("/web/runs/vae/vae-001/metrics", params={"since": 0}).json()
    series = {s["key"]: s["points"] for s in body["series"]}
    assert series["loss"] == [[1, 0.8]]


def test_config_and_summary(client):
    assert client.get("/web/runs/vae/vae-001/config").json()["config"]["batch_size"] == 64
    assert client.get("/web/runs/vae/vae-001/summary").json()["summary"]["status"] == "completed"


def test_artifact_served(client):
    r = client.get("/web/runs/vae/vae-001/artifacts/sample.png")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/png")


def test_unknown_run_404(client):
    assert client.get("/web/runs/nope/nope/metrics").status_code == 404


def test_models_endpoint(client):
    from mini_networks.core.registry import MODEL_NAMES

    names = {m["name"] for m in client.get("/web/models").json()}
    assert names == set(MODEL_NAMES)
    sample = client.get("/web/models").json()[0]
    assert "config_schema" in sample and "defaults" in sample
