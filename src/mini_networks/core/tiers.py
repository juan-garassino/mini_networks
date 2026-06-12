"""Tier budget table — the single visible place training budgets live.

BaseConfig.effective_* properties read from here. `None` means uncapped.
Per-model overrides are added during Phase 2 stabilization triage; each
override carries a comment saying why that model needs a different budget.
"""
from __future__ import annotations

DEFAULTS: dict[str, dict[str, int | None]] = {
    #      epochs  batch_cap  sample_limit  train_batches  eval_batches  timesteps
    "S": {"epochs": 1, "batch_cap": 16, "sample_limit": 32,
          "train_batches": 1, "eval_batches": 1, "timesteps": 25},
    "M": {"epochs": 3, "batch_cap": 32, "sample_limit": 512,
          "train_batches": 8, "eval_batches": 4, "timesteps": 200},
    "L": {"epochs": None, "batch_cap": None, "sample_limit": None,
          "train_batches": None, "eval_batches": None, "timesteps": None},
}

# model_name -> tier -> partial budget override (merged over DEFAULTS)
MODEL_OVERRIDES: dict[str, dict[str, dict[str, int | None]]] = {}


def budget(model_name: str, tier: str, key: str) -> int | None:
    override = MODEL_OVERRIDES.get(model_name, {}).get(tier, {})
    if key in override:
        return override[key]
    return DEFAULTS[tier][key]
