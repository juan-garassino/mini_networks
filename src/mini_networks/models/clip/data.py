"""Image-text pair dataset for CLIP: pairs MNIST images with digit label captions."""
from __future__ import annotations

import random
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Caption templates — multiple per class for richer text supervision.
# Randomly sampling one per training step acts as caption augmentation.
# ---------------------------------------------------------------------------

_DIGIT_NAMES = ["zero", "one", "two", "three", "four",
                 "five", "six", "seven", "eight", "nine"]


def _build_captions(label: int) -> list[str]:
    name = _DIGIT_NAMES[label]
    n = str(label)
    return [
        name,
        f"digit {name}",
        f"number {name}",
        f"numeral {n}",
        f"a handwritten {name}",
        f"the number {n}",
        f"a photo of the digit {name}",
        f"written {name}",
        f"a clear {name}",
        f"handwritten number {n}",
        f"the digit {n}",
        f"a bold {name}",
    ]


# Dict: class id → list[str] caption pool
DIGIT_CAPTIONS: dict[int, list[str]] = {i: _build_captions(i) for i in range(10)}


# ---------------------------------------------------------------------------
# Tokenisation helpers (char-level: ord(c) % vocab_size)
# ---------------------------------------------------------------------------

def _text_to_ids(text: str, seq_len: int, vocab_size: int) -> list[int]:
    ids = [ord(c) % vocab_size for c in text]
    if len(ids) < seq_len:
        ids = ids + [0] * (seq_len - len(ids))
    return ids[:seq_len]


def label_to_tokens(label: int, seq_len: int = 32, vocab_size: int = 256) -> torch.Tensor:
    """Sample a random caption for *label* and return its token tensor [seq_len]."""
    text = random.choice(DIGIT_CAPTIONS[label])
    return torch.tensor(_text_to_ids(text, seq_len, vocab_size), dtype=torch.long)


def label_to_all_tokens(label: int, seq_len: int = 32, vocab_size: int = 256) -> torch.Tensor:
    """Return all caption token tensors for *label* as [N, seq_len] batch.

    Useful for averaging embeddings across the full caption pool.
    """
    rows = [_text_to_ids(cap, seq_len, vocab_size) for cap in DIGIT_CAPTIONS[label]]
    return torch.tensor(rows, dtype=torch.long)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MNISTImageTextDataset(Dataset):
    """Each item is (image [1,28,28], tokens [seq_len], label int).

    Tokens are sampled randomly from the caption pool for *label* — this
    acts as caption augmentation so the model sees diverse text paired
    with each image.
    """

    def __init__(
        self,
        data_root: str,
        train: bool = True,
        seq_len: int = 32,
        vocab_size: int = 256,
        fast_demo: bool = False,
    ):
        self._ds = torchvision.datasets.MNIST(
            root=data_root, train=train, download=True, transform=T.ToTensor()
        )
        self._seq_len = seq_len
        self._vocab_size = vocab_size
        self._limit = 256 if fast_demo else len(self._ds)

    def __len__(self) -> int:
        return self._limit

    def __getitem__(self, idx: int):
        image, label = self._ds[idx]
        tokens = label_to_tokens(int(label), self._seq_len, self._vocab_size)
        return image, tokens, int(label)
