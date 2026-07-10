"""Dataset registry with MNIST task modes + text loaders."""
from __future__ import annotations

import contextlib
import io
import os
import urllib.request
import wave
from typing import Literal

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

import logging

log = logging.getLogger(__name__)

TaskMode = Literal[
    "classification",
    "binary_segmentation",
    "multiclass_segmentation",
    "detection",
    "clip",
    "contrastive",
]

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
_PREPARED_DATASETS: set[tuple[str, str]] = set()


def _prepare_dataset_once(name: str, data_root: str) -> None:
    key = (name, os.path.abspath(data_root))
    if key not in _PREPARED_DATASETS:
        log.info(f"Preparing {name}...")


def _mark_dataset_ready(name: str, data_root: str) -> None:
    key = (name, os.path.abspath(data_root))
    if key not in _PREPARED_DATASETS:
        _PREPARED_DATASETS.add(key)
        log.info(f"{name} ready")


def _load_torchvision_dataset(dataset_cls, data_root: str, train: bool, transform, name: str | None = None):
    dataset_name = name or dataset_cls.__name__
    _prepare_dataset_once(dataset_name, data_root)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ds = dataset_cls(
            root=data_root,
            train=train,
            download=True,
            transform=transform,
        )
    _mark_dataset_ready(dataset_name, data_root)
    return ds


class MNISTClassification(Dataset):
    """Standard MNIST for classification: returns (image [1,28,28], label int)."""

    def __init__(self, data_root: str, train: bool = True, fast_demo: bool = False):
        ds = _load_torchvision_dataset(
            torchvision.datasets.MNIST,
            data_root=data_root,
            train=train,
            transform=T.ToTensor(),
            name="MNIST",
        )
        self._data = ds
        self._limit = 256 if fast_demo else len(ds)

    def __len__(self) -> int:
        return self._limit

    def __getitem__(self, idx: int):
        return self._data[idx]


class BinarySegmentationFromDigits(Dataset):
    """Digit dataset where target is a binary pixel mask (foreground vs background)."""

    def __init__(
        self,
        dataset_cls,
        data_root: str,
        train: bool = True,
        fast_demo: bool = False,
    ):
        ds = _load_torchvision_dataset(
            dataset_cls,
            data_root=data_root,
            train=train,
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


class MulticlassSegmentationFromDigits(Dataset):
    """
    Two digit images overlaid; target is 12-class segmentation mask.
    Classes 0-9: digit pixels (exclusive), 10: background, 11: intersection.
    """

    def __init__(
        self,
        dataset_cls,
        data_root: str,
        train: bool = True,
        fast_demo: bool = False,
    ):
        ds = _load_torchvision_dataset(
            dataset_cls,
            data_root=data_root,
            train=train,
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


class DigitDetection(Dataset):
    """
    MNIST digit placed randomly on a 56x56 canvas.
    Target: (label int, bbox [x1, y1, x2, y2] normalized to [0,1]).
    """

    def __init__(
        self,
        dataset_cls,
        data_root: str,
        train: bool = True,
        canvas_size: int = 56,
        fast_demo: bool = False,
    ):
        ds = _load_torchvision_dataset(
            dataset_cls,
            data_root=data_root,
            train=train,
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


# ---------------------------------------------------------------------------
# Contrastive pair dataset (SimCLR-style)
# ---------------------------------------------------------------------------

class ContrastivePairFromDigits(Dataset):
    """Return two augmented views of the same digit image."""

    def __init__(
        self,
        dataset_cls,
        data_root: str,
        train: bool = True,
        fast_demo: bool = False,
        image_size: int = 28,
    ):
        ds = _load_torchvision_dataset(
            dataset_cls,
            data_root=data_root,
            train=train,
            transform=T.ToTensor(),
        )
        self._data = ds
        self._limit = 256 if fast_demo else len(ds)
        self._aug = T.Compose(
            [
                T.ToPILImage(),
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                T.RandomRotation(20),
                T.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return self._limit

    def __getitem__(self, idx: int):
        image, label = self._data[idx]
        v1 = self._aug(image)
        v2 = self._aug(image)
        return v1, v2, int(label)


# ---------------------------------------------------------------------------
# Synthetic audio + tabular datasets
# ---------------------------------------------------------------------------

class SyntheticAudioDigits(Dataset):
    """Synthetic sine waves for 10 classes (0-9). Returns (waveform, label)."""

    def __init__(
        self,
        n_samples: int = 1000,
        sample_len: int = 256,
        n_classes: int = 10,
        fast_demo: bool = False,
        seed: int = 123,
    ):
        self.n_classes = n_classes
        self.sample_len = sample_len
        self._limit = 256 if fast_demo else n_samples
        g = torch.Generator().manual_seed(seed)
        self._labels = torch.randint(0, n_classes, (self._limit,), generator=g)

    def __len__(self) -> int:
        return self._limit

    def __getitem__(self, idx: int):
        label = int(self._labels[idx])
        freq = 1.0 + label * 0.2
        t = torch.linspace(0, 1, self.sample_len)
        wave = torch.sin(2 * torch.pi * freq * t)
        wave = wave + 0.05 * torch.randn_like(wave)
        return wave.unsqueeze(0), label  # [1, T]


class SyntheticTabular(Dataset):
    """Simple Gaussian blobs. Returns (features, label)."""

    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int = 8,
        n_classes: int = 3,
        fast_demo: bool = False,
        seed: int = 123,
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self._limit = 256 if fast_demo else n_samples
        g = torch.Generator().manual_seed(seed)
        centers = torch.randn(n_classes, n_features, generator=g) * 2.0
        labels = torch.randint(0, n_classes, (self._limit,), generator=g)
        data = centers[labels] + 0.5 * torch.randn(self._limit, n_features, generator=g)
        self._data = data
        self._labels = labels

    def __len__(self) -> int:
        return self._limit

    def __getitem__(self, idx: int):
        return self._data[idx], int(self._labels[idx])


class SpeechDigitsDataset(Dataset):
    """Free Spoken Digit Dataset (FSDD): 0-9 spoken digits (wav)."""

    FSDD_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"

    def __init__(
        self,
        data_root: str,
        fast_demo: bool = False,
        sample_len: int = 4000,
        require_downloads: bool = True,
    ):
        self.data_root = data_root
        self.sample_len = sample_len
        self._files = _ensure_fsdd(data_root, require_downloads=require_downloads)
        if fast_demo:
            self._files = self._files[: min(50, len(self._files))]

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int):
        path = self._files[idx]
        label = int(os.path.basename(path).split("_")[0])
        wave_tensor = _read_wav_mono(path, self.sample_len)
        return wave_tensor, label


class IrisDataset(Dataset):
    """Iris dataset (4 features, 3 classes)."""

    IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    def __init__(self, data_root: str, fast_demo: bool = False, require_downloads: bool = True):
        self.data_root = data_root
        X, y = _ensure_iris(data_root, require_downloads=require_downloads)
        if fast_demo:
            X = X[:50]
            y = y[:50]
        self._X = torch.tensor(X, dtype=torch.float32)
        # Standardize with DATASET-global stats at load. Per-batch normalization
        # downstream broke eval: iris.data is class-sorted, so unshuffled eval
        # batches are near-single-class and batch stats erase the class signal
        # (accuracy pinned at 0.55 regardless of training — m-triage-3).
        self._X = (self._X - self._X.mean(dim=0)) / (self._X.std(dim=0) + 1e-6)
        self._y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self._y)

    def __getitem__(self, idx: int):
        return self._X[idx], int(self._y[idx])

# ---------------------------------------------------------------------------
# CLIP-style MNIST image-text dataset
# ---------------------------------------------------------------------------

_DIGIT_NAMES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]


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


DIGIT_CAPTIONS: dict[int, list[str]] = {i: _build_captions(i) for i in range(10)}


def _text_to_ids(text: str, seq_len: int, vocab_size: int) -> list[int]:
    ids = [ord(c) % vocab_size for c in text]
    if len(ids) < seq_len:
        ids = ids + [0] * (seq_len - len(ids))
    return ids[:seq_len]


def label_to_tokens(label: int, seq_len: int = 32, vocab_size: int = 256) -> torch.Tensor:
    """Sample a random caption for *label* and return its token tensor [seq_len]."""
    import random

    text = random.choice(DIGIT_CAPTIONS[label])
    return torch.tensor(_text_to_ids(text, seq_len, vocab_size), dtype=torch.long)


def label_to_all_tokens(label: int, seq_len: int = 32, vocab_size: int = 256) -> torch.Tensor:
    """Return all caption token tensors for *label* as [N, seq_len] batch."""
    rows = [_text_to_ids(cap, seq_len, vocab_size) for cap in DIGIT_CAPTIONS[label]]
    return torch.tensor(rows, dtype=torch.long)


class MNISTImageTextDataset(Dataset):
    """Each item is (image [1,28,28], tokens [seq_len], label int)."""

    def __init__(
        self,
        data_root: str,
        train: bool = True,
        seq_len: int = 32,
        vocab_size: int = 256,
        fast_demo: bool = False,
    ):
        self._ds = _load_torchvision_dataset(
            torchvision.datasets.MNIST,
            data_root=data_root,
            train=train,
            transform=T.ToTensor(),
            name="MNIST",
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


class TextFileDataset(Dataset):
    """Simple character-level text dataset from a single file."""

    def __init__(self, file_path: str, seq_len: int = 128, fast_demo: bool = False):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        if fast_demo:
            text = text[:4096]
        self.file_path = file_path
        self.text = text
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
        self.file_path = file_path
        self.text = text

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

    Supported names: "mnist", "fashion_mnist", "text_file", "tiny_shakespeare",
    "synthetic_audio_digits", "synthetic_tabular", "speech_digits", "iris"
    """
    train = split == "train"
    os.makedirs(data_root, exist_ok=True)

    if name in ("mnist", "fashion_mnist"):
        cls = torchvision.datasets.MNIST if name == "mnist" else torchvision.datasets.FashionMNIST
        dataset_name = "MNIST" if name == "mnist" else "FashionMNIST"
        if task == "classification":
            ds = _load_torchvision_dataset(
                cls,
                data_root=data_root,
                train=train,
                transform=T.ToTensor(),
                name=dataset_name,
            )
            if fast_demo:
                # Wrap to limit size
                return _Subset(ds, 256)
            return ds
        elif task == "clip":
            if name != "mnist":
                raise ValueError("CLIP dataset only supported for MNIST digits.")
            return MNISTImageTextDataset(
                data_root=data_root,
                train=train,
                seq_len=kwargs.get("seq_len", 32),
                vocab_size=kwargs.get("vocab_size", 256),
                fast_demo=fast_demo,
            )
        elif task == "binary_segmentation":
            return BinarySegmentationFromDigits(
                cls,
                data_root=data_root,
                train=train,
                fast_demo=fast_demo,
            )
        elif task == "multiclass_segmentation":
            return MulticlassSegmentationFromDigits(
                cls,
                data_root=data_root,
                train=train,
                fast_demo=fast_demo,
            )
        elif task == "detection":
            canvas_size = kwargs.get("canvas_size", 56)
            return DigitDetection(
                cls,
                data_root=data_root,
                train=train,
                canvas_size=canvas_size,
                fast_demo=fast_demo,
            )
        elif task == "contrastive":
            return ContrastivePairFromDigits(
                cls,
                data_root=data_root,
                train=train,
                fast_demo=fast_demo,
                image_size=kwargs.get("image_size", 28),
            )
        else:
            raise ValueError(f"Unknown task: {task}")
    elif name in ("text_file", "tiny_shakespeare"):
        file_path = kwargs.get("file_path") or kwargs.get("text_file")
        if name == "tiny_shakespeare" or not file_path:
            file_path = _ensure_tiny_shakespeare(data_root)
        seq_len = kwargs.get("seq_len", 128)
        tokenizer_type = kwargs.get("tokenizer_type", "char")
        bpe_vocab_size = kwargs.get("bpe_vocab_size", 512)
        if tokenizer_type == "bpe":
            return BPETextFileDataset(
                file_path=file_path,
                seq_len=seq_len,
                bpe_vocab_size=bpe_vocab_size,
                fast_demo=fast_demo,
            )
        return TextFileDataset(file_path=file_path, seq_len=seq_len, fast_demo=fast_demo)
    elif name == "synthetic_audio_digits":
        return SyntheticAudioDigits(
            n_samples=kwargs.get("n_samples", 1000),
            sample_len=kwargs.get("sample_len", 256),
            n_classes=kwargs.get("n_classes", 10),
            fast_demo=fast_demo,
            seed=kwargs.get("seed", 123),
        )
    elif name == "synthetic_tabular":
        return SyntheticTabular(
            n_samples=kwargs.get("n_samples", 1000),
            n_features=kwargs.get("n_features", 8),
            n_classes=kwargs.get("n_classes", 3),
            fast_demo=fast_demo,
            seed=kwargs.get("seed", 123),
        )
    elif name == "speech_digits":
        return SpeechDigitsDataset(
            data_root=data_root,
            fast_demo=fast_demo,
            sample_len=kwargs.get("sample_len", 4000),
            require_downloads=kwargs.get("require_downloads", True),
        )
    elif name == "iris":
        return IrisDataset(
            data_root=data_root,
            fast_demo=fast_demo,
            require_downloads=kwargs.get("require_downloads", True),
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_dataloader(
    name: str,
    data_root: str,
    split: Literal["train", "val"] = "train",
    task: TaskMode = "classification",
    batch_size: int = 32,
    fast_demo: bool = False,
    sample_limit: int | None = None,
    **kwargs,
) -> DataLoader:
    ds = get_dataset(name, data_root, split=split, task=task, fast_demo=fast_demo, **kwargs)
    if sample_limit is not None:
        ds = _Subset(ds, sample_limit)
    shuffle = split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


def _ensure_tiny_shakespeare(data_root: str) -> str:
    """Download Tiny Shakespeare into data_root if missing. Returns file path."""
    os.makedirs(data_root, exist_ok=True)
    path = os.path.join(data_root, "shakespeare.txt")
    _prepare_dataset_once("Tiny Shakespeare", data_root)
    if not os.path.exists(path):
        log.info("Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    _mark_dataset_ready("Tiny Shakespeare", data_root)
    return path


def _ensure_fsdd(data_root: str, require_downloads: bool = True) -> list[str]:
    import zipfile

    os.makedirs(data_root, exist_ok=True)
    zip_path = os.path.join(data_root, "fsdd.zip")
    extract_dir = os.path.join(data_root, "fsdd")
    wav_dir = os.path.join(extract_dir, "free-spoken-digit-dataset-master", "recordings")
    _prepare_dataset_once("FSDD", data_root)

    if not os.path.exists(wav_dir):
        if not os.path.exists(zip_path):
            if not require_downloads:
                raise RuntimeError("FSDD not found and downloads disabled.")
            log.info("Downloading FSDD...")
            urllib.request.urlretrieve(SpeechDigitsDataset.FSDD_URL, zip_path)
        log.info("Extracting FSDD...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    files = [
        os.path.join(wav_dir, f)
        for f in os.listdir(wav_dir)
        if f.endswith(".wav")
    ]
    files.sort()
    if not files:
        raise RuntimeError("FSDD download failed or no wav files found.")
    _mark_dataset_ready("FSDD", data_root)
    return files


def _read_wav_mono(path: str, sample_len: int) -> torch.Tensor:
    with wave.open(path, "rb") as wf:
        n = wf.getnframes()
        raw = wf.readframes(n)
        # 16-bit PCM little endian
        import numpy as np

        audio = np.frombuffer(raw, dtype=np.int16).astype("float32") / 32768.0
        if wf.getnchannels() > 1:
            audio = audio.reshape(-1, wf.getnchannels()).mean(axis=1)
        # Pad or crop
        if len(audio) < sample_len:
            pad = sample_len - len(audio)
            audio = np.pad(audio, (0, pad))
        else:
            audio = audio[:sample_len]
        return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)


def _ensure_iris(data_root: str, require_downloads: bool = True):
    os.makedirs(data_root, exist_ok=True)
    path = os.path.join(data_root, "iris.data")
    _prepare_dataset_once("Iris", data_root)
    if not os.path.exists(path):
        if not require_downloads:
            raise RuntimeError("Iris not found and downloads disabled.")
        log.info("Downloading Iris...")
        urllib.request.urlretrieve(IrisDataset.IRIS_URL, path)

    X = []
    y = []
    mapping = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2,
    }
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 5:
                continue
            feats = list(map(float, parts[:4]))
            label = mapping.get(parts[4], None)
            if label is None:
                continue
            X.append(feats)
            y.append(label)
    if not X:
        raise RuntimeError("Iris download failed or dataset is empty.")
    _mark_dataset_ready("Iris", data_root)
    return X, y


class _Subset(Dataset):
    """Seeded RANDOM subset — not a head-slice. Head-slicing biased
    class-sorted datasets: FSDD's first ~512 files are almost all digits 0-1,
    which let the audio classifiers score a fake 1.0 at the old M sample
    budget and collapse on the honest full set (m-full-2)."""

    def __init__(self, dataset: Dataset, limit: int, seed: int = 123):
        self._dataset = dataset
        self._limit = min(limit, len(dataset))
        g = torch.Generator().manual_seed(seed)
        self._indices = torch.randperm(len(dataset), generator=g)[: self._limit].tolist()

    def __len__(self) -> int:
        return self._limit

    def __getitem__(self, idx: int):
        return self._dataset[self._indices[idx]]

    def __getattr__(self, name: str):
        return getattr(self._dataset, name)


def make_classification_dataloader(config, split: str = "train") -> DataLoader:
    """Standard image-classification loader driven entirely by the config.

    Shared by the supervised vision trainers (classifier, resnet, vit,
    mobilenet, convnext) — their make_*_dataloader names alias this.
    """
    return get_dataloader(
        name=config.dataset,
        data_root=config.data_root,
        split=split,
        task="classification",
        batch_size=config.effective_batch_size,
        fast_demo=config.effective_fast_demo,
        sample_limit=config.dataset_sample_limit,
    )
