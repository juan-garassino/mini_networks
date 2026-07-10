"""Task sharding for the ONE parallel Cloud Run Job gate sweep.

One job execution fans the full item list (models + compositions, catalog
order) across tasks: task i (``CLOUD_RUN_TASK_INDEX``) runs the quality gate
for item i and writes its CheckResult as a JSON shard to
``gs://$MN_SWEEP_BUCKET/$MN_SWEEP_PREFIX/sweeps/$SWEEP_ID/shards/<item>.json``
(plus a local copy under checkpoint_root). ``merge_sweep_report`` collects the
shards into one report.{md,json} with the same writers and exit semantics as
``sweep --check``; items whose shard never arrived (task crashed before
upload) are reported as missing and fail the merge.

GCS upload is skipped when ``MN_SWEEP_BUCKET`` is unset (local dry-runs);
imports are lazy so the base install never needs google-cloud-storage.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

log = logging.getLogger(__name__)

BUCKET_ENV = "MN_SWEEP_BUCKET"
PREFIX_ENV = "MN_SWEEP_PREFIX"
DEFAULT_BUCKET = "garassino-ml-artifacts"
DEFAULT_PREFIX = "mini-networks"


def default_items() -> list[str]:
    from mini_networks.colab.catalog import COMPOSITIONS, MODELS

    return list(MODELS) + list(COMPOSITIONS)


def resolve_items(raw: str | None) -> list[str]:
    """ITEMS env/flag (comma list) or the full canonical list. Order matters:
    the task index maps into this list, so caller and container must agree."""
    if not raw:
        return default_items()
    items = [x.strip() for x in raw.split(",") if x.strip()]
    known = set(default_items())
    unknown = [x for x in items if x not in known]
    if unknown:
        raise ValueError(f"Unknown items: {', '.join(unknown)}")
    return items


def _shards_prefix(prefix: str, sweep_id: str) -> str:
    return f"{prefix}/sweeps/{sweep_id}/shards"


def _upload(bucket_name: str, blob_path: str, local: Path) -> None:
    from google.cloud import storage

    storage.Client().bucket(bucket_name).blob(blob_path).upload_from_filename(str(local))


def run_sweep_task(args: argparse.Namespace) -> int:
    """Gate ONE item selected by CLOUD_RUN_TASK_INDEX; write its shard."""
    from mini_networks.colab.catalog import MODELS
    from mini_networks.colab.gate import JudgeContext, check_composition, check_model

    items = resolve_items(os.environ.get("ITEMS"))
    index = args.index if args.index is not None else int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))
    if index >= len(items):
        log.info("task index %d >= %d items — nothing to do", index, len(items))
        return 0
    item = items[index]
    sweep_id = os.environ.get("SWEEP_ID", "adhoc")
    log.info("sweep task %d/%d: %s (tier %s, sweep %s)", index, len(items), item, args.training_tier, sweep_id)

    gate_args = argparse.Namespace(
        fast_demo=False,
        training_tier=args.training_tier,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        fail_fast=False,
    )
    if item in MODELS:
        result = check_model(item, gate_args, JudgeContext(gate_args))
    else:
        result = check_composition(item, gate_args)

    payload = {
        "meta": {
            "sweep_id": sweep_id,
            "index": index,
            "tier": args.training_tier,
            "device": args.device,
        },
        "result": asdict(result),
    }
    local = Path(args.checkpoint_root) / "sweep-shards" / sweep_id / f"{item}.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))

    bucket = os.environ.get(BUCKET_ENV)
    if bucket:
        prefix = os.environ.get(PREFIX_ENV, DEFAULT_PREFIX)
        blob_path = f"{_shards_prefix(prefix, sweep_id)}/{item}.json"
        _upload(bucket, blob_path, local)
        log.info("shard uploaded: gs://%s/%s", bucket, blob_path)
    else:
        log.info("%s unset — shard kept local only: %s", BUCKET_ENV, local)

    log.info("item %s: %s", item, result.status)
    return 0 if result.status == "pass" else 1


def build_merged_report(
    payloads: list[dict], expected: list[str], sweep_id: str, out_dir: str | Path
) -> tuple[Path, Path, bool]:
    """Pure merge: shards + expected item list → report.{md,json} + ok flag."""
    from mini_networks.core.sweep_report import CheckResult, write_report

    results = [CheckResult(**p["result"]) for p in payloads]
    by_name = {r.name: r for r in results}
    ordered = [by_name[n] for n in expected if n in by_name]
    ordered += [r for r in results if r.name not in set(expected)]
    missing = sorted(set(expected) - set(by_name))

    tiers = {p["meta"].get("tier") for p in payloads}
    meta = {
        "sweep_id": sweep_id,
        "tier": "/".join(sorted(t for t in tiers if t)) or "unknown",
        "items_expected": len(expected),
        "items_reported": len(ordered),
        "missing_shards": missing,
    }
    md_path, json_path = write_report(ordered, out_dir, meta)
    ok = not missing and all(r.status == "pass" for r in ordered)
    return md_path, json_path, ok


def merge_sweep_report(args: argparse.Namespace) -> int:
    """Download all shards of a sweep from GCS, merge, upload the report back."""
    from google.cloud import storage

    expected = resolve_items(args.items)
    client = storage.Client()
    prefix = _shards_prefix(args.prefix, args.sweep_id)
    payloads = [
        json.loads(blob.download_as_bytes())
        for blob in client.list_blobs(args.bucket, prefix=prefix + "/")
        if blob.name.endswith(".json")
    ]
    out_dir = Path(args.out_root) / "sweep" / args.sweep_id
    md_path, json_path, ok = build_merged_report(payloads, expected, args.sweep_id, out_dir)

    bucket = client.bucket(args.bucket)
    for local in (md_path, json_path):
        bucket.blob(f"{args.prefix}/sweeps/{args.sweep_id}/{local.name}").upload_from_filename(str(local))

    print(md_path.read_text())
    print(f"Report: {md_path} (mirrored to gs://{args.bucket}/{args.prefix}/sweeps/{args.sweep_id}/)")
    return 0 if ok else 1
