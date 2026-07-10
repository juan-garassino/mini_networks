"""Pure metric helpers: pivot, since-filter, tail."""
from __future__ import annotations

from mini_networks.web.metrics import pivot_long_to_series, tail_latest

ROWS = [
    {"step": 0, "key": "loss", "value": 1.5},
    {"step": 0, "key": "acc", "value": 0.2},
    {"step": 1, "key": "loss", "value": 0.9},
    {"step": 1, "key": "acc", "value": 0.5},
    {"step": 2, "key": "loss", "value": 0.4},
    {"step": 0, "key": "note", "value": "string-dropped"},
]


def test_pivot_groups_by_key_sorted():
    series = dict(pivot_long_to_series(ROWS))
    assert series["loss"] == [(0, 1.5), (1, 0.9), (2, 0.4)]
    assert series["acc"] == [(0, 0.2), (1, 0.5)]
    assert "note" not in series  # non-numeric dropped


def test_since_filters_strictly_greater():
    series = dict(pivot_long_to_series(ROWS, since=0))
    assert series["loss"] == [(1, 0.9), (2, 0.4)]


def test_tail_latest():
    step, latest = tail_latest(ROWS)
    assert step == 2
    assert latest == {"loss": 0.4}


def test_empty():
    assert pivot_long_to_series([]) == []
    assert tail_latest([]) == (None, {})
