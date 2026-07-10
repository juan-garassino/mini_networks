"""MN_DISABLE_TRAIN=1 hard-disables both training entry points (public showcase)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from mini_networks.api.main import app


def test_train_endpoint_rejected(monkeypatch):
    monkeypatch.setenv("MN_DISABLE_TRAIN", "1")
    r = TestClient(app).post("/train/classifier", json={})
    assert r.status_code == 403
    assert "disabled" in r.json()["detail"]


def test_compose_endpoint_rejected(monkeypatch):
    monkeypatch.setenv("MN_DISABLE_TRAIN", "1")
    r = TestClient(app).post("/compose/image_captioning", json={})
    assert r.status_code == 403


def test_inference_still_allowed(monkeypatch):
    monkeypatch.setenv("MN_DISABLE_TRAIN", "1")
    r = TestClient(app).get("/infer/classifier/info")
    assert r.status_code == 200
