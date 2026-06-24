"""LocalRunsSource over a fixture tree + Local/MLflow parity."""
from __future__ import annotations

import pytest

from mini_networks.web.sources import LocalRunsSource, RunNotFound


def test_lists_all_runs_including_nested(runs_dir):
    src = LocalRunsSource(str(runs_dir))
    by_id = {r.id: r for r in src.list_runs()}
    assert set(by_id) == {"vae/vae-001", "gan/gan-001", "clip/212716/212716"}
    assert by_id["vae/vae-001"].status == "done"
    assert by_id["gan/gan-001"].status == "running"  # state, no summary
    assert by_id["clip/212716/212716"].model == "clip"


def test_summary_fields(runs_dir):
    src = LocalRunsSource(str(runs_dir))
    vae = next(r for r in src.list_runs() if r.id == "vae/vae-001")
    assert vae.last_step == 1
    assert vae.last_metrics == {"loss": 0.8, "epoch": 1.0}
    assert sorted(vae.artifact_names) == ["model.pt", "sample.png"]


def test_metrics_pivot_and_config(runs_dir):
    src = LocalRunsSource(str(runs_dir))
    m = src.get_metrics("vae/vae-001")
    series = {s.key: s.points for s in m.series}
    assert series["loss"] == [(0, 1.5), (1, 0.8)]
    assert m.latest_step == 1
    assert src.get_config("vae/vae-001").config == {"lr": 0.001, "batch_size": 64}


def test_unknown_run_raises(runs_dir):
    with pytest.raises(RunNotFound):
        LocalRunsSource(str(runs_dir)).get_metrics("nope/nope")


def test_artifact_resolution_is_sandboxed(runs_dir):
    src = LocalRunsSource(str(runs_dir))
    path, media = src.open_artifact("vae/vae-001", "sample.png")
    assert path.name == "sample.png" and media == "image/png"
    with pytest.raises(RunNotFound):
        src.open_artifact("vae/vae-001", "../../config.yaml")  # path traversal blocked


def test_stale_run_without_summary_is_done(tmp_path):
    import os
    import time

    base = tmp_path / "runs"
    d = base / "mamba" / "mamba-old"
    (d / "artifacts").mkdir(parents=True)
    (d / "metrics.jsonl").write_text('{"step": 0, "key": "loss", "value": 1.0}\n')
    old = time.time() - 3600
    os.utime(d / "metrics.jsonl", (old, old))

    r = next(x for x in LocalRunsSource(str(base)).list_runs() if x.id == "mamba/mamba-old")
    assert r.status == "done"  # not "running" — process is long gone


def test_local_mlflow_parity(monkeypatch, tmp_path):
    """A run logged through the Logger MLflow sink reads back identically."""
    pytest.importorskip("mlflow")
    monkeypatch.setenv("MN_MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("MN_MLFLOW_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MN_MLFLOW_EXPERIMENT", "parity")

    from mini_networks.core.logging.logger import Logger
    from mini_networks.web.sources import MLflowSource

    runs = tmp_path / "runs"
    logger = Logger(output_dir=str(runs / "vae"), run_name="vae-001")
    logger.log_config({"lr": 0.001})
    logger.log_metrics(0, {"loss": 1.5})
    logger.log_metrics(1, {"loss": 0.8})
    logger.close()

    local = LocalRunsSource(str(runs)).get_metrics("vae/vae-001")
    ml_src = MLflowSource(experiment="parity")
    run_id = ml_src.list_runs()[0].id
    remote = ml_src.get_metrics(run_id)

    local_series = {s.key: s.points for s in local.series}
    remote_series = {s.key: s.points for s in remote.series}
    assert local_series == remote_series == {"loss": [(0, 1.5), (1, 0.8)]}
