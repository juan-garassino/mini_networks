"""Registry champion loader: local checkpoints for every Production mini-<model>.

The cloud gate registers gate-passing M/L checkpoints as ``mini-<model>``
versions with champion/challenger promotion. This module pulls each model's
current Production champion into ``<checkpoint_root>/champions/<model>/artifacts/``
— the exact layout ``BaseTrainer.load_checkpoint`` expects — so the playground's
inference endpoints can serve every cloud-trained model locally:

    python main.py pull-champions          # all models
    python main.py serve                   # /infer/<model> now uses champions

``POST /infer/{model}`` falls back to the champion checkpoint automatically
when the request names no run_id/checkpoint. Requires the ``cloud`` extra and
GCS read access (ADC) because the tracker hands out direct gs:// artifact URIs.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from mini_networks.core.logging.mlflow_registry import MODEL_PREFIX
from mini_networks.core.logging.mlflow_sink import TRACKING_URI_ENV

log = logging.getLogger(__name__)

CHAMPIONS_DIRNAME = "champions"


def champion_artifacts_dir(model_name: str, checkpoint_root: str | Path) -> Path:
    return Path(checkpoint_root) / CHAMPIONS_DIRNAME / model_name / "artifacts"


def has_champion(model_name: str, checkpoint_root: str | Path) -> bool:
    d = champion_artifacts_dir(model_name, checkpoint_root)
    return d.is_dir() and any(d.glob("*.pt"))


def _make_client():
    # Tiny factory so tests can stub the client; lazy import keeps base light.
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=os.environ[TRACKING_URI_ENV])


def pull_champions(models: list[str] | None = None, checkpoint_root: str | Path = "runs") -> dict[str, str]:
    """Download each model's Production champion. Returns {model: status} where
    status is 'vN' (pulled), 'no champion', or 'error: …' — never raises."""
    from mini_networks.core.registry import MODEL_NAMES

    if not os.environ.get(TRACKING_URI_ENV):
        return {m: "error: MN_MLFLOW_TRACKING_URI unset" for m in (models or MODEL_NAMES)}
    client = _make_client()
    status: dict[str, str] = {}
    for name in models or MODEL_NAMES:
        try:
            versions = client.get_latest_versions(f"{MODEL_PREFIX}{name}", stages=["Production"])
            if not versions:
                status[name] = "no champion"
                continue
            mv = versions[0]
            dest = champion_artifacts_dir(name, checkpoint_root)
            dest.mkdir(parents=True, exist_ok=True)
            staging = dest.parent / "_dl"
            staging.mkdir(parents=True, exist_ok=True)  # mlflow requires dst_path to exist
            local = Path(client.download_artifacts(mv.run_id, "model", dst_path=str(staging)))
            pulled = 0
            for f in local.glob("*.pt"):
                shutil.copy2(f, dest / f.name)
                pulled += 1
            shutil.rmtree(staging, ignore_errors=True)
            if not pulled:
                status[name] = "error: champion has no *.pt artifacts"
                continue
            (dest.parent / "VERSION").write_text(f"{MODEL_PREFIX}{name} v{mv.version} run {mv.run_id}\n")
            status[name] = f"v{mv.version}"
            log.info("champion %s: %s v%s (%d file(s))", name, MODEL_PREFIX + name, mv.version, pulled)
        except Exception as e:
            # A missing registered model surfaces as an exception in some
            # mlflow versions — report it as absence, keep real errors loud.
            if "RESOURCE_DOES_NOT_EXIST" in str(e) or "not found" in str(e).lower():
                status[name] = "no champion"
            else:
                status[name] = f"error: {type(e).__name__}: {e}"
                log.warning("champion %s failed: %s", name, e)
    return status
