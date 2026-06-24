"""Registry introspection for the playground — shared by /web/models and /infer/info."""
from __future__ import annotations

from typing import Any

from mini_networks.core.registry import MODEL_NAMES, get_model_registry


def build_model_info(name: str) -> dict[str, Any]:
    """Config JSON-schema + defaults for one model (drives Lab/Sandbox forms)."""
    registry = get_model_registry()
    ConfigClass, _, _ = registry[name]
    return {
        "name": name,
        "family": None,
        "config_schema": ConfigClass.model_json_schema(),
        "defaults": ConfigClass().model_dump(),
    }


def list_model_infos() -> list[dict[str, Any]]:
    registry = get_model_registry()
    return [build_model_info(n) for n in MODEL_NAMES if n in registry]
