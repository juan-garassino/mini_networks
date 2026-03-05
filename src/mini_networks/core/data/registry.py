"""Dataset registry with MNIST in 4 task modes + text loader."""
from __future__ import annotations

import os
from typing import Literal, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T

from mini_networks.core.data.transforms import (
    make_binary_mask,
    make_composite_image,
    make_multiclass_mask,
    place_on_canvas,
)

TaskMode = Literal["classification", "binary_segmentation", "multiclass_segmentation", "detection"]


class MNISTClassification(Dataset):
    """Standard MNIST for classification: returns (image [1,28,28], label int)."""

    def __init__(self, data_root: str, train: bool = True, fast_demo: bool = False):
        ds = torchvision.datasets.MNIST(
            root=data_root, train=train, download=True,
            transform=T.ToTensor(),
        )
        self._data = ds
        self._limit = 256 if fast_demo else len(ds)

    def __len__(self) -> int:
        return self._limit

    def __getitem__(self, idx: int):
        return self._data[idx]


class MNISTBinarySegmentation(Dataset):
    """MNIST where target is a binary pixel mask (foreground vs background)."""

    def __init__(self, data_root: str, train: bool = True, fast_demo: bool = False):
        ds = torchvision.datasets.MNIST(
            root=data_root, train=train, download=True,
            transform=T.ToTensor(),
        )
        self._data = ds
        self._limit = 256 if fast_demo else len(ds)

    def __len__(self) -> int:
        return self._limit

    def __getitem__(self, idx: int):
        image, _ = self._data[idx]
        mask = make_binary_mask(image, threshold=0.0)
        return image, mask


class MNISTMulticlassSegmentation(Dataset):
    """
    Two MNIST digits overlaid; target is 12-class segmentation mask.
    Classes 0-9: digit pixels (exclusive), 10: background, 11: intersection.
    """

    def __init__(self, data_root: str, train: bool = True, fast_demo: bool = False):
        ds = torchvision.datasets.MNIST(
            root=data_root, train=train, download=True,
            transform=T.ToTensor(),
        )
        self._data = ds
        n = len(ds) // 2
        self._limit = 128 if fast_demo else n

    def __len__(self) -> int:
        return self._limit

    def __getitem__(self, idx: int):
        img_a, label_a = self._data[idx * 2]
        img_b, label_b = self._data[idx * 2 + 1]
        composite = make_composite_image(img_a, img_b)
        mask = make_multiclass_mask(img_a, int(label_a), img_b, int(label_b))
        return composite, mask


class MNISTDetection(Dataset):
    """
    MNIST digit placed randomly on a 56x56 canvas.
    Target: (label int, bbox [x1, y1, x2, y2] normalized to [0,1]).
    """

    def __init__(
        self,
        data_root: str,
        train: bool = True,
        canvas_size: int = 56,
        fast_demo: bool = False,
    ):
        ds = torchvision.datasets.MNIST(
            root=data_root, train=train, download=True,
            transform=T.ToTensor(),
        )
        self._data = ds
        self._canvas_size = canvas_size
        self._limit = 256 if fast_demo else len(ds)

    def __len__(self) -> int:
        return self._limit

    def __getitem__(self, idx: int):
        image, label = self._data[idx]
        canvas, bbox = place_on_canvas(image, self._canvas_size)
        bbox_norm = [c / self._canvas_size for c in bbox]
        return canvas, int(label), torch.tensor(bbox_norm, dtype=torch.float32)


class TextFileDataset(Dataset):
    """Simple character-level text dataset from a single file."""

    def __init__(self, file_path: str, seq_len: int = 128, fast_demo: bool = False):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        if fast_demo:
            text = text[:4096]
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self._seq_len = seq_len
        self._data = data

    @property
    def tokenizer(self):
        """Return a CharTokenizer built from this dataset's vocabulary."""
        from mini_networks.models.transformer.tokenizer import CharTokenizer
        tok = CharTokenizer(self.stoi)
        tok.itos = self.itos
        return tok

    def __len__(self) -> int:
        return max(1, len(self._data) - self._seq_len)

    def __getitem__(self, idx: int):
        x = self._data[idx: idx + self._seq_len]
        y = self._data[idx + 1: idx + self._seq_len + 1]
        return x, y


class BPETextFileDataset(Dataset):
    """Text dataset tokenized with BPE (byte-level pair encoding)."""

    def __init__(
        self,
        file_path: str,
        seq_len: int = 128,
        bpe_vocab_size: int = 512,
        fast_demo: bool = False,
    ):
        from mini_networks.models.transformer.tokenizer import BPETokenizer

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        if fast_demo:
            text = text[:8192]

        tok = BPETokenizer()
        tok.train(text, vocab_size=bpe_vocab_size, min_frequency=2)
        self._tokenizer = tok

        ids = tok.encode(text)
        self._data = torch.tensor(ids, dtype=torch.long)
        self._seq_len = seq_len

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    def __len__(self) -> int:
        return max(1, len(self._data) - self._seq_len)

    def __getitem__(self, idx: int):
        x = self._data[idx: idx + self._seq_len]
        y = self._data[idx + 1: idx + self._seq_len + 1]
        return x, y


def get_dataset(
    name: str,
    data_root: str,
    split: Literal["train", "val"] = "train",
    task: TaskMode = "classification",
    fast_demo: bool = False,
    **kwargs,
) -> Dataset:
    """
    Factory returning a Dataset for the given name and task mode.

    Supported names: "mnist", "fashion_mnist", "text_file"
    """
    train = split == "train"
    os.makedirs(data_root, exist_ok=True)

    if name in ("mnist", "fashion_mnist"):
        cls = torchvision.datasets.MNIST if name == "mnist" else torchvision.datasets.FashionMNIST
        if task == "classification":
            ds = cls(root=data_root, train=train, download=True, transform=T.ToTensor())
            if fast_demo:
                # Wrap to limit size
                return _Subset(ds, 256)
            return ds
        elif task == "binary_segmentation":
            return MNISTBinarySegmentation(data_root, train=train, fast_demo=fast_demo)
        elif task == "multiclass_segmentation":
            return MNISTMulticlassSegmentation(data_root, train=train, fast_demo=fast_demo)
        elif task == "detection":
            canvas_size = kwargs.get("canvas_size", 56)
            return MNISTDetection(data_root, train=train, canvas_size=canvas_size, fast_demo=fast_demo)
        else:
            raise ValueError(f"Unknown task: {task}")
    elif name == "text_file":
        file_path = kwargs.get("file_path") or kwargs.get("data_root", data_root)
        seq_len = kwargs.get("seq_len", 128)
        return TextFileDataset(file_path=file_path, seq_len=seq_len, fast_demo=fast_demo)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_dataloader(
    name: str,
    data_root: str,
    split: Literal["train", "val"] = "train",
    task: TaskMode = "classification",
    batch_size: int = 32,
    fast_demo: bool = False,
    **kwargs,
) -> DataLoader:
    ds = get_dataset(name, data_root, split=split, task=task, fast_demo=fast_demo, **kwargs)
    shuffle = split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


class _Subset(Dataset):
    def __init__(self, dataset: Dataset, limit: int):
        self._dataset = dataset
        self._limit = min(limit, len(dataset))

    def __len__(self) -> int:
        return self._limit

    def __getitem__(self, idx: int):
        return self._dataset[idx]
