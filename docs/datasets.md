# Datasets and Task Modes

The shared data registry lives in `src/mini_networks/core/data/registry.py`. It supports consistent data loading across models, compositions, and Colab.

**Vision Datasets**
- `mnist`
- `fashion_mnist`

**Vision Task Modes**
- `classification` returns `(image, label)` for single‑digit classification.
- `binary_segmentation` returns `(image, mask)` with foreground/background.
- `multiclass_segmentation` returns `(composite_image, mask)` where two digits are overlaid.
- `detection` returns `(canvas, label, bbox)` with a digit placed on a larger canvas.
- `clip` returns `(image, tokens, label)` for MNIST image‑text contrastive learning.
- `contrastive` returns `(view1, view2, label)` for SimCLR‑style pretraining.

**Text Datasets**
- `text_file` uses an explicit file path if provided.
- `tiny_shakespeare` auto‑downloads and caches Tiny Shakespeare.

**Audio and Tabular (Synthetic)**
- `synthetic_audio_digits` sine‑wave digits (0–9), returns `(waveform, label)`.
- `synthetic_tabular` Gaussian blobs, returns `(features, label)`.

**Audio and Tabular (Real)**
- `speech_digits` Free Spoken Digit Dataset (FSDD), returns `(waveform, label)`.
- `iris` UCI Iris dataset, returns `(features, label)`.

**Disable Downloads**
For real datasets, pass `require_downloads=False` to `get_dataloader()` if you want to avoid network access. A clean error is raised if files are missing.

**Tokenizer Types**
The registry can build either character or BPE tokenizers:
- `tokenizer_type=char`
- `tokenizer_type=bpe` with `bpe_vocab_size`

**Why This Matters**
Many models reuse the same dataset with different supervision targets. For example:
- MNIST classification and segmentation share the same raw digits.
- FashionMNIST can be used for both classification and derived segmentation.
- Tiny Shakespeare drives Transformer, RNN, Mamba, RAG, and RLHF.
