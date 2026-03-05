"""Smoke tests for the FastAPI layer.

Tests use FastAPI's synchronous TestClient (no running server needed).

WARNING: ``@pytest.mark.slow`` tests actually run full training loops via
Starlette's TestClient (which executes BackgroundTasks synchronously).
They are skipped by default.  Run them on a fast machine with:

    uv run pytest tests/test_api.py -m slow

or include all tests with:

    uv run pytest tests/test_api.py -m "slow or not slow"
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mini_networks.api.main import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Model info (GET /infer/{model}/info)
# ---------------------------------------------------------------------------

class TestModelInfo:
    @pytest.mark.parametrize("model", ["clip", "transformer", "rl_maze", "rlhf"])
    def test_info_200(self, client, model):
        r = client.get(f"/infer/{model}/info")
        assert r.status_code == 200, r.text

    def test_info_returns_schema_and_defaults(self, client):
        r = client.get("/infer/clip/info")
        body = r.json()
        assert "model" in body
        assert "config_schema" in body
        assert "defaults" in body
        assert body["model"] == "clip"

    def test_info_unknown_model_404(self, client):
        r = client.get("/infer/nonexistent_model/info")
        assert r.status_code == 404

    @pytest.mark.parametrize("model", ["diffusion", "gan", "rnn", "lora", "rag"])
    def test_info_additional_models(self, client, model):
        r = client.get(f"/infer/{model}/info")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# List runs (GET /train/)
# ---------------------------------------------------------------------------

class TestListRuns:
    def test_list_initially_empty_or_list(self, client):
        r = client.get("/train/")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


# ---------------------------------------------------------------------------
# Start training (POST /train/{model})
# Marked @pytest.mark.slow — TestClient executes BackgroundTasks synchronously,
# so each call actually runs full training. Skip on slow machines.
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestStartTraining:
    def _fast_body(self):
        return {
            "epochs": 1,
            "batch_size": 4,
            "fast_demo": True,
            "data_root": "/tmp/mini_networks_test_data",
            "device": "cpu",
        }

    def test_train_clip_starts(self, client):
        r = client.post("/train/clip", json=self._fast_body())
        assert r.status_code == 200
        body = r.json()
        assert "job_id" in body
        assert body["status"] == "started"
        assert "output_dir" in body

    def test_train_transformer_starts(self, client):
        r = client.post("/train/transformer", json=self._fast_body())
        assert r.status_code == 200
        assert r.json()["status"] == "started"

    def test_train_gan_starts(self, client):
        r = client.post("/train/gan", json=self._fast_body())
        assert r.status_code == 200

    def test_train_rl_maze_starts(self, client):
        r = client.post("/train/rl_maze", json=self._fast_body())
        assert r.status_code == 200

    def test_train_rlhf_starts(self, client):
        r = client.post("/train/rlhf", json=self._fast_body())
        assert r.status_code == 200

    def test_train_unknown_model_404(self, client):
        r = client.post("/train/no_such_model", json=self._fast_body())
        assert r.status_code == 404

    @pytest.mark.parametrize("model", ["diffusion", "rnn", "lora", "rag", "mamba"])
    def test_train_additional_models_start(self, client, model):
        r = client.post(f"/train/{model}", json=self._fast_body())
        assert r.status_code == 200, f"{model}: {r.text}"

    def test_job_id_format(self, client):
        r = client.post("/train/clip", json=self._fast_body())
        job_id = r.json()["job_id"]
        # Expected format: "clip-YYYYMMDD-HHMMSS"
        assert job_id.startswith("clip-")

    def test_train_with_extra_params(self, client):
        body = self._fast_body()
        body["extra"] = {"n_layers": 1, "n_heads": 2}
        r = client.post("/train/transformer", json=body)
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Job status (GET /train/{job_id}/status)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestJobStatus:
    def test_status_after_start(self, client):
        body = {
            "epochs": 1,
            "batch_size": 4,
            "fast_demo": True,
            "data_root": "/tmp/mini_networks_test_data",
            "device": "cpu",
        }
        r = client.post("/train/rnn", json=body)
        job_id = r.json()["job_id"]

        r2 = client.get(f"/train/{job_id}/status")
        assert r2.status_code == 200
        status_body = r2.json()
        assert status_body["job_id"] == job_id
        assert status_body["model"] == "rnn"
        assert status_body["status"] in ("pending", "running", "done", "failed")

    def test_status_unknown_job_404(self, client):
        r = client.get("/train/nonexistent-job-id-xyz/status")
        assert r.status_code == 404

    def test_list_shows_started_jobs(self, client):
        # Start a job
        body = {
            "epochs": 1,
            "batch_size": 4,
            "fast_demo": True,
            "data_root": "/tmp/mini_networks_test_data",
        }
        r = client.post("/train/lora", json=body)
        job_id = r.json()["job_id"]

        r2 = client.get("/train/")
        jobs = r2.json()
        ids = [j["job_id"] for j in jobs]
        assert job_id in ids


# ---------------------------------------------------------------------------
# Inference (POST /infer/{model})
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestInference:
    def test_infer_transformer_returns_outputs(self, client):
        body = {"prompt": "KING", "n_samples": 1, "inputs": {"max_new_tokens": 8}}
        r = client.post("/infer/transformer", json=body)
        assert r.status_code == 200
        resp = r.json()
        assert resp["model"] == "transformer"
        assert "outputs" in resp
        assert isinstance(resp["outputs"], dict)

    def test_infer_rlhf_returns_outputs(self, client):
        body = {"prompt": "thou", "n_samples": 1, "inputs": {"max_new_tokens": 8}}
        r = client.post("/infer/rlhf", json=body)
        assert r.status_code == 200

    def test_infer_rl_maze_returns_outputs(self, client):
        body = {"inputs": {}}
        r = client.post("/infer/rl_maze", json=body)
        assert r.status_code == 200

    def test_infer_unknown_model_404(self, client):
        r = client.post("/infer/no_such_model", json={"inputs": {}})
        assert r.status_code == 404

    def test_infer_rnn_returns_outputs(self, client):
        body = {"prompt": "hello", "n_samples": 1, "inputs": {"max_new_tokens": 8}}
        r = client.post("/infer/rnn", json=body)
        assert r.status_code == 200

    def test_infer_response_schema(self, client):
        r = client.post("/infer/transformer", json={"prompt": "A", "inputs": {}})
        body = r.json()
        assert set(body.keys()) >= {"model", "outputs"}
        assert body["model"] == "transformer"
