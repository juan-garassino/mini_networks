"""Sweep sharding: item resolution, out-of-range tasks, and shard merging.

Pure-unit coverage — the gate itself is exercised by the S-tier check sweep in
CI; here we test the index→item contract and the merge/missing-shard logic
without touching GCS or torch training.
"""
from __future__ import annotations

import argparse
import json

import pytest

from mini_networks.cloud import sweep_shard
from mini_networks.colab.catalog import COMPOSITIONS, MODELS


def _payload(name, status="pass", tier="M", item_type="model"):
    return {
        "meta": {"sweep_id": "s1", "tier": tier, "index": 0, "device": "cpu"},
        "result": {"item_type": item_type, "name": name, "status": status, "tier": tier},
    }


class TestResolveItems:
    def test_default_is_full_catalog_in_order(self):
        assert sweep_shard.resolve_items(None) == list(MODELS) + list(COMPOSITIONS)

    def test_parses_comma_list(self):
        assert sweep_shard.resolve_items("classifier, gan") == ["classifier", "gan"]

    def test_unknown_item_raises(self):
        with pytest.raises(ValueError, match="nope"):
            sweep_shard.resolve_items("classifier,nope")

    def test_dino_is_in_the_catalog(self):
        assert "dino" in sweep_shard.default_items()


class TestRunSweepTask:
    def test_index_out_of_range_is_a_noop_success(self, monkeypatch):
        monkeypatch.setenv("ITEMS", "classifier")
        monkeypatch.delenv("CLOUD_RUN_TASK_INDEX", raising=False)
        args = argparse.Namespace(index=5, training_tier="S", epochs=1, batch_size=8,
                                  device="cpu", data_root="/tmp", checkpoint_root="/tmp")
        assert sweep_shard.run_sweep_task(args) == 0


class TestBuildMergedReport:
    def test_all_pass(self, tmp_path):
        payloads = [_payload("classifier"), _payload("gan")]
        md, js, ok = sweep_shard.build_merged_report(payloads, ["classifier", "gan"], "s1", tmp_path)
        assert ok
        report = json.loads(js.read_text())
        assert report["meta"]["missing_shards"] == []
        assert [r["name"] for r in report["results"]] == ["classifier", "gan"]

    def test_failure_flips_ok(self, tmp_path):
        payloads = [_payload("classifier"), _payload("gan", status="fail")]
        _, _, ok = sweep_shard.build_merged_report(payloads, ["classifier", "gan"], "s1", tmp_path)
        assert not ok

    def test_missing_shard_detected_and_fails(self, tmp_path):
        payloads = [_payload("classifier")]
        md, js, ok = sweep_shard.build_merged_report(payloads, ["classifier", "gan"], "s1", tmp_path)
        assert not ok
        assert json.loads(js.read_text())["meta"]["missing_shards"] == ["gan"]

    def test_orders_by_expected_list(self, tmp_path):
        payloads = [_payload("gan"), _payload("classifier")]
        _, js, _ = sweep_shard.build_merged_report(payloads, ["classifier", "gan"], "s1", tmp_path)
        assert [r["name"] for r in json.loads(js.read_text())["results"]] == ["classifier", "gan"]
