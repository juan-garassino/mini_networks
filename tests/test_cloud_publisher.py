"""Cloud publisher + the Lab→cloud branch in POST /train."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mini_networks.cloud.publisher import JobSpec, NullPublisher, get_publisher


def test_jobspec_roundtrip():
    spec = JobSpec(model="vae", tier="M", hparams={"epochs": 3}, run_name="vae-1")
    assert JobSpec.model_validate_json(spec.model_dump_json()) == spec


def test_get_publisher_null_when_unconfigured(monkeypatch):
    monkeypatch.delenv("MN_PUBSUB_TOPIC", raising=False)
    pub = get_publisher()
    assert isinstance(pub, NullPublisher)
    with pytest.raises(RuntimeError):
        pub.publish(JobSpec(model="vae", run_name="x"))


def test_cloud_branch_publishes_and_dispatches(monkeypatch, tmp_path):
    monkeypatch.setenv("MINI_NETWORKS_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("MN_TRAIN_BACKEND", "cloud")

    published = []

    class FakePublisher:
        def publish(self, spec):
            published.append(spec)
            return "msg-1"

    import mini_networks.api.routers.training as tr

    monkeypatch.setattr(tr, "get_publisher", lambda: FakePublisher())

    from mini_networks.api.main import create_app

    client = TestClient(create_app())
    r = client.post("/train/vae", json={"fast_demo": True, "training_tier": "S", "extra": {"latent_dim": 8}})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "dispatched"

    assert len(published) == 1
    spec = published[0]
    assert spec.model == "vae" and spec.tier == "S" and spec.run_name == body["job_id"]
    assert spec.hparams["latent_dim"] == 8

    # Dispatched stub is visible; no local training ran.
    assert client.get(f"/train/{body['job_id']}/status").json()["status"] == "dispatched"
    assert not (tmp_path / "runs" / "vae" / body["job_id"] / "metrics.jsonl").exists()


def test_local_backend_still_default(monkeypatch):
    monkeypatch.delenv("MN_TRAIN_BACKEND", raising=False)
    from mini_networks.api.routers.training import _train_backend

    assert _train_backend() == "local"
