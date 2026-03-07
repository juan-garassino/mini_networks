"""Smoke tests for the shared data registry (MNIST/Fashion + text)."""
import os

from mini_networks.core.data.registry import get_dataloader

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


def _next_batch(dl):
    for batch in dl:
        return batch
    raise AssertionError("Dataloader returned no batches")


def test_mnist_tasks_smoke():
    for task in ["classification", "binary_segmentation", "multiclass_segmentation", "detection", "clip", "contrastive"]:
        dl = get_dataloader(
            name="mnist",
            data_root=DATA_ROOT,
            split="train",
            task=task,
            batch_size=4,
            fast_demo=True,
            seq_len=16,
            vocab_size=64,
        )
        batch = _next_batch(dl)
        assert batch is not None


def test_fashion_tasks_smoke():
    for task in ["classification", "binary_segmentation", "multiclass_segmentation", "detection", "contrastive"]:
        dl = get_dataloader(
            name="fashion_mnist",
            data_root=DATA_ROOT,
            split="train",
            task=task,
            batch_size=4,
            fast_demo=True,
        )
        batch = _next_batch(dl)
        assert batch is not None


def test_text_datasets_smoke():
    dl = get_dataloader(
        name="tiny_shakespeare",
        data_root=DATA_ROOT,
        split="train",
        batch_size=4,
        fast_demo=True,
        seq_len=32,
    )
    batch = _next_batch(dl)
    assert batch is not None


def test_synthetic_audio_smoke():
    dl = get_dataloader(
        name="synthetic_audio_digits",
        data_root=DATA_ROOT,
        split="train",
        batch_size=4,
        fast_demo=True,
        sample_len=256,
    )
    batch = _next_batch(dl)
    assert batch is not None


def test_synthetic_tabular_smoke():
    dl = get_dataloader(
        name="synthetic_tabular",
        data_root=DATA_ROOT,
        split="train",
        batch_size=4,
        fast_demo=True,
        n_features=8,
    )
    batch = _next_batch(dl)
    assert batch is not None


def test_speech_digits_smoke():
    dl = get_dataloader(
        name="speech_digits",
        data_root=DATA_ROOT,
        split="train",
        batch_size=2,
        fast_demo=True,
        sample_len=4000,
        require_downloads=False,
    )
    batch = _next_batch(dl)
    assert batch is not None


def test_iris_smoke():
    dl = get_dataloader(
        name="iris",
        data_root=DATA_ROOT,
        split="train",
        batch_size=4,
        fast_demo=True,
        require_downloads=False,
    )
    batch = _next_batch(dl)
    assert batch is not None
