"""Shared test helpers.

Some datasets (FSDD speech digits, Iris) are fetched from the network on first
use; tests run with require_downloads=False so an absent local cache raises
RuntimeError("... downloads disabled"). Those tests must skip, not fail —
CI has no dataset cache either.
"""
from __future__ import annotations

import contextlib

import pytest


@contextlib.contextmanager
def dataset_or_skip():
    try:
        yield
    except RuntimeError as exc:
        if "downloads disabled" in str(exc):
            pytest.skip(str(exc))
        raise


@pytest.fixture
def skip_if_dataset_missing():
    return dataset_or_skip
