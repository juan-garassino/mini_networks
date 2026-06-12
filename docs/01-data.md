# 01 — Data: one registry, many tasks

Every model in this repo gets its data from the same factory. No trainer downloads or
parses anything itself — it calls the registry, which returns a standard PyTorch
`Dataset` or `DataLoader`. This is what makes models interchangeable: the same MNIST
images feed a classifier, a segmenter, a detector, a CLIP pair, and a diffusion model,
each through a different *task mode*.

## The registry

`src/mini_networks/core/data/registry.py` exposes two functions:

```python
get_dataset(name, data_root, split="train", task="classification", fast_demo=False, **kwargs)
get_dataloader(name, data_root, ..., batch_size=32, sample_limit=None, **kwargs)
```

Supported names: `mnist`, `fashion_mnist`, `tiny_shakespeare`, `text_file`,
`synthetic_audio_digits`, `synthetic_tabular`, `speech_digits`, `iris`.
`fast_demo=True` caps datasets at ~256 samples; `sample_limit` (set by the tier
system, see `docs/00-overview.md`) wraps the dataset in a `_Subset`.

## The 4 MNIST task modes

One source dataset, four supervision signals — derived on the fly in
`core/data/transforms.py`:

| `task` | Item shape | Target |
|---|---|---|
| `classification` | image `(1,28,28)` | `int` label 0–9 |
| `binary_segmentation` | image `(1,28,28)` | `(28,28)` long mask `{0,1}` (digit pixels vs background) |
| `multiclass_segmentation` | composite of **two** overlaid digits `(1,28,28)` | `(28,28)` long mask `{0..11}` — classes 0–9 digit pixels, 10 background, 11 intersection |
| `detection` | digit placed randomly on a `(1,56,56)` canvas | `int` label + `(4,)` bbox `[x1,y1,x2,y2]` normalized to `[0,1]` |

Two extra modes ride on the same digits: `clip` returns
`(image, caption_tokens[seq_len], label)` where captions are templated digit phrases
("a handwritten seven", ...), and `contrastive` returns two augmented views of the
same image (random resized crop + rotation) for SimCLR-style training.

## Text

- `tiny_shakespeare` auto-downloads Karpathy's `input.txt` into `data_root/shakespeare.txt`.
- `TextFileDataset` — char-level: builds vocab from the file's character set, items are
  `(x[seq_len], y[seq_len])` next-token pairs. Exposes a `.tokenizer` (`CharTokenizer`).
- `BPETextFileDataset` — same file tokenized with the repo's own byte-level
  `BPETokenizer` (`models/transformer/tokenizer.py`), default `bpe_vocab_size=512`,
  trained on the fly. Select with `tokenizer_type="bpe"`.

Both feed the transformer, RNN, Mamba, RAG, and RLHF trainers unchanged — only
`vocab_size` differs.

## Audio

- `synthetic_audio_digits` — sine waves at class-dependent frequency plus noise,
  `(waveform[1,256], label)`. Zero downloads; used in fast tests.
- `speech_digits` — the Free Spoken Digit Dataset (FSDD): real spoken-digit `.wav`
  files downloaded as a zip from GitHub, read with stdlib `wave`, padded/cropped to
  `sample_len` (default 4000), returned as `(waveform[1,T], label)`.

The four audio models share this loader and differ only in the representation computed
*inside the trainer* (`core/data/audio.py`): raw waveform 1D CNN (`audio_classifier`),
STFT magnitude 2D CNN (`audio_spectrogram`), mel-spectrogram 2D CNN
(`audio_melspectrogram`), and a transformer over spectrogram frames
(`audio_transformer`).

## Tabular

- `synthetic_tabular` — Gaussian blobs, 8 features, 3 classes, seeded.
- `iris` — the UCI Iris CSV (4 features, 3 classes), downloaded once into
  `data_root/iris.data` and parsed by hand.

## Download gating

Datasets that hit the network on first use (FSDD, Iris) accept
`require_downloads: bool = True`. With `require_downloads=False` and no local cache
they raise `RuntimeError("... downloads disabled")` instead of fetching. Tests always
pass `require_downloads=False`; `tests/conftest.py` provides the `dataset_or_skip`
context manager, which converts that specific `RuntimeError` into `pytest.skip`. So
the suite is deterministic offline: dataset-dependent tests skip on a cold machine
(CI has no cache) and run when the cache exists. MNIST/FashionMNIST go through
torchvision's own `download=True` path with stdout silenced; tiny shakespeare always
downloads if missing (it is a single small text file).
