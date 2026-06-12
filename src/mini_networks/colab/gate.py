"""Quality-gate runner behind `main.py sweep --check`.

Per item: train at the requested tier, run S-style sanity checks on the
metrics series, round-trip the checkpoint through a fresh trainer, compute
the EvalSpec metric on the loaded weights (M/L gate on its threshold), and
probe inference. Crashes become status="error" with a traceback tail; the
sweep keeps going unless --fail-fast.
"""
from __future__ import annotations

import logging
import math
import time
import traceback
from pathlib import Path
from typing import Any

from mini_networks.core.evalspec import EvalSpec, get_eval_spec, passes_threshold
from mini_networks.core.sweep_report import CheckResult

log = logging.getLogger(__name__)

JUDGED_MODELS = {"diffusion", "gan", "pixelcnn"}


# ---------------------------------------------------------------------------
# S-tier sanity checks on the metrics.jsonl series
# ---------------------------------------------------------------------------

def _series(metrics: list[dict], key: str) -> list[float]:
    return [
        m["value"] for m in metrics
        if m.get("key") == key and isinstance(m.get("value"), (int, float))
    ]


def loss_series_check(metrics: list[dict], spec: EvalSpec) -> tuple[bool, dict]:
    """Finiteness on every loss series; downward trend on at least one.

    Single-point series (S tier logs once per epoch, 1 epoch) cannot show a
    trend — reported as skipped, never faked as a pass or fail.
    """
    info: dict[str, Any] = {"keys": [], "finite": True, "trend": "no series"}
    found_any = False
    trend_ok = False
    trend_checkable = False
    for key in spec.loss_keys:
        series = _series(metrics, key)
        if not series:
            continue
        found_any = True
        info["keys"].append(key)
        if not all(math.isfinite(v) for v in series):
            info["finite"] = False
            info["trend"] = f"non-finite values in '{key}'"
            return False, info
        if len(series) >= 2:
            trend_checkable = True
            if series[-1] < series[0]:
                trend_ok = True
    if not found_any:
        info["trend"] = f"no loss series found (keys tried: {list(spec.loss_keys)})"
        return False, info
    if spec.s_mode == "finite":
        info["trend"] = "skipped (s_mode=finite)"
        return True, info
    if not trend_checkable:
        info["trend"] = "skipped (single point)"
        return True, info
    info["trend"] = "ok" if trend_ok else "no loss decreased (last >= first on all keys)"
    return trend_ok, info


# ---------------------------------------------------------------------------
# Judge scorer: rate generative samples with a trained MNIST classifier
# ---------------------------------------------------------------------------

class JudgeContext:
    """Lazily trains/loads the classifier used to score generative samples."""

    def __init__(self, args):
        self._args = args
        self._trainer = None
        self._config = None
        self.run_dir: str | None = None

    def get(self):
        if self._trainer is not None:
            return self._trainer, self._config
        import torch  # noqa: F401
        from mini_networks.core.registry import get_model_registry
        from mini_networks.core.checkpoints import latest_run_dir
        from mini_networks.colab.runners import run_model, _run_base

        args = self._args
        ConfigClass, TrainerClass, _ = get_model_registry()["classifier"]
        tier = "S" if args.fast_demo else args.training_tier
        config = ConfigClass(
            epochs=args.epochs, batch_size=args.batch_size, fast_demo=args.fast_demo,
            training_tier=tier, data_root=args.data_root, device=args.device,
            checkpoint_root=args.checkpoint_root, resume=False,
        )
        run = latest_run_dir(_run_base(args.checkpoint_root, "classifier"))
        if run is None or not (run / "artifacts" / "model.pt").exists():
            log.info("judge: no classifier checkpoint found, training one")
            logger = run_model(
                "classifier", epochs=args.epochs, batch_size=args.batch_size,
                fast_demo=args.fast_demo, training_tier=tier, data_root=args.data_root,
                device=args.device, checkpoint_root=args.checkpoint_root,
                resume=False, validate_inference=False,
            )
            run = logger.run_dir
        trainer = TrainerClass()
        trainer.load_checkpoint(config, run / "artifacts")
        self._trainer, self._config, self.run_dir = trainer, config, str(run)
        return trainer, config


def judge_samples(trainer, config, judge: JudgeContext, args) -> float:
    """judge_score = mean max-softmax confidence x digit-class coverage."""
    import torch

    judge_trainer, judge_config = judge.get()
    n = 16 if config.effective_tier == "S" else 64
    out = trainer.infer(config, {"n_samples": n, "seed": config.seed})
    samples = out["samples"] if isinstance(out, dict) else out
    samples = samples.to(judge_config.device).float()
    if samples.dim() == 3:
        samples = samples.unsqueeze(1)
    with torch.no_grad():
        logits = judge_trainer.model(samples)
        probs = torch.softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values.mean().item()
    coverage = len(set(probs.argmax(dim=-1).tolist())) / probs.shape[-1]
    log.info("judge: confidence=%.3f coverage=%.3f", confidence, coverage)
    return confidence * coverage


SCORE_FNS = {"judge_samples": judge_samples}


# ---------------------------------------------------------------------------
# Per-item checks
# ---------------------------------------------------------------------------

def _extract_metric(metric_key: str, eval_metrics: dict, jsonl_metrics: list[dict]) -> float | None:
    value = eval_metrics.get(metric_key)
    if isinstance(value, (int, float)):
        return float(value)
    series = _series(jsonl_metrics, metric_key)
    return float(series[-1]) if series else None


def _decide(result: CheckResult, spec: EvalSpec, s_ok: bool, tier: str) -> None:
    if not s_ok:
        result.status = "fail"
        return
    if spec.roundtrip and result.roundtrip != "ok":
        result.status = "fail"
        return
    threshold = spec.thresholds.get(tier)
    if threshold is not None:
        result.threshold = threshold
        if result.value is None:
            result.status = "error"
            result.error = (
                f"metric {spec.metric!r} not found in evaluate() output nor metrics.jsonl"
            )
            return
        if not passes_threshold(result.value, threshold, spec.higher_is_better):
            result.status = "fail"
            return
    result.status = "pass"


def check_model(name: str, args, judge: JudgeContext) -> CheckResult:
    from mini_networks.core.registry import get_model_registry
    from mini_networks.colab.runners import run_model
    from mini_networks.colab.probes import _run_model_inference_probe

    spec = get_eval_spec(name)
    tier = "S" if args.fast_demo else args.training_tier
    result = CheckResult(item_type="model", name=name, status="error", tier=tier,
                         metric=spec.metric, higher_is_better=spec.higher_is_better)
    t0 = time.perf_counter()
    try:
        logger = run_model(
            name, epochs=args.epochs, batch_size=args.batch_size,
            fast_demo=args.fast_demo, training_tier=tier, data_root=args.data_root,
            device=args.device, checkpoint_root=args.checkpoint_root,
            resume=False, validate_inference=False,
        )
        result.run_dir = str(logger.run_dir)

        ConfigClass, TrainerClass, dataloader_fn = get_model_registry()[name]
        config = ConfigClass(
            epochs=args.epochs, batch_size=args.batch_size, fast_demo=args.fast_demo,
            training_tier=tier, data_root=args.data_root, device=args.device,
            checkpoint_root=args.checkpoint_root, resume=False,
        ).model_copy(update={"run_name": logger.run_dir.name,
                             "output_dir": str(logger.run_dir)})

        s_ok, result.s_check = loss_series_check(logger.read_metrics(), spec)

        fresh = TrainerClass()
        try:
            fresh.load_checkpoint(config, logger.artifacts_dir)
            result.roundtrip = "ok"
        except Exception as exc:
            result.roundtrip = f"failed: {exc}"
        target = fresh if result.roundtrip == "ok" else None

        if spec.metric is not None and target is not None:
            if spec.score_fn is not None:
                result.value = SCORE_FNS[spec.score_fn](target, config, judge, args)
            else:
                try:
                    dl = dataloader_fn(config, split="test")
                except Exception:
                    dl = dataloader_fn(config, split="train")
                eval_metrics = target.evaluate(config, dl, logger)
                result.value = _extract_metric(spec.metric, eval_metrics, logger.read_metrics())

        if target is not None:
            dl_train = dataloader_fn(config, split="train")
            result.infer_summary = _run_model_inference_probe(name, target, config, dl_train)

        _decide(result, spec, s_ok, tier)
    except Exception:
        result.status = "error"
        result.error = "\n".join(traceback.format_exc().splitlines()[-15:])
    result.duration_s = time.perf_counter() - t0
    return result


def check_composition(name: str, args) -> CheckResult:
    from mini_networks.colab.runners import run_composition
    from mini_networks.colab.probes import _validate_probe_output
    from mini_networks.core.logging.logger import Logger

    spec = get_eval_spec(name)
    tier = "S" if args.fast_demo else args.training_tier
    result = CheckResult(item_type="composition", name=name, status="error", tier=tier,
                         metric=spec.metric, higher_is_better=spec.higher_is_better)
    t0 = time.perf_counter()
    try:
        output = run_composition(
            name, fast_demo=args.fast_demo, training_tier=tier,
            data_root=args.data_root, device=args.device,
            checkpoint_root=args.checkpoint_root, validate_inference=False,
        )
        run_dir = output.get("run_dir") if isinstance(output, dict) else None
        result.run_dir = run_dir

        metrics: list[dict] = []
        if run_dir:
            metrics = Logger(output_dir=run_dir, run_name=Path(run_dir).name).read_metrics()
        s_ok, result.s_check = loss_series_check(metrics, spec)

        result.infer_summary = _validate_probe_output(output)

        if spec.metric is not None and isinstance(output, dict):
            result.value = _extract_metric(spec.metric, output, metrics)

        _decide(result, spec, s_ok, tier)
    except Exception:
        result.status = "error"
        result.error = "\n".join(traceback.format_exc().splitlines()[-15:])
    result.duration_s = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def run_checked_sweep(models: list[str], compositions: list[str], args) -> tuple[list[CheckResult], dict]:
    judge = JudgeContext(args)

    # Judge-scored models need the classifier trained first.
    if any(m in JUDGED_MODELS for m in models) and "classifier" in models:
        models = ["classifier"] + [m for m in models if m != "classifier"]

    results: list[CheckResult] = []
    for name in models:
        log.info("checking model %s", name)
        result = check_model(name, args, judge)
        results.append(result)
        if result.status != "pass" and args.fail_fast:
            break
    else:
        for name in compositions:
            log.info("checking composition %s", name)
            result = check_composition(name, args)
            results.append(result)
            if result.status != "pass" and args.fail_fast:
                break

    meta = {
        "tier": "S" if args.fast_demo else args.training_tier,
        "device": args.device,
        "models": len(models),
        "compositions": len(compositions),
        "judge_run_dir": judge.run_dir,
    }
    return results, meta
