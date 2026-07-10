# Chapter 09 — CLIP and Multimodal Learning

## Theory recap

CLIP learns a **shared embedding space** for two modalities. An image encoder
and a text encoder each map their input to a vector; training pulls matched
(image, caption) pairs together and pushes mismatched pairs apart. The loss is
**symmetric contrastive** (InfoNCE both ways): for a batch of B pairs, build
the B×B similarity matrix, treat the diagonal as the correct class, and
average cross-entropy over rows (image→text) and columns (text→image). A
temperature scales the logits — sharper similarity distributions as it grows.
Once the space is trained, zero-shot classification is just "which caption
embedding is closest to this image embedding?".

## In this repo

- `src/mini_networks/models/clip/model.py` — `CLIPModel` holds two encoders:
  `ImageEncoder` (3-layer CNN with BatchNorm → adaptive pool → linear
  projection) and `TextEncoder` (token + positional embeddings →
  `nn.TransformerEncoder` with a padding mask → mean-pool over non-pad
  positions → projection). Both outputs are L2-normalized in
  `encode_image()` / `encode_text()`.
- `contrastive_loss()` implements the symmetric loss with a **learnable
  temperature**: `log_temperature = nn.Parameter(log(1/0.07))`, logits are
  `image_embeds @ text_embeds.T * temperature`, labels are `arange(B)`, and
  the result is `(loss_i + loss_t) / 2`.
- Captions come from templates in `src/mini_networks/core/data/registry.py`:
  `_build_captions(label)` produces 12 variants per digit ("seven",
  "digit seven", "a handwritten seven", "a photo of the digit seven", ...),
  collected in `DIGIT_CAPTIONS`. `label_to_tokens()` samples one at random
  per training example and encodes it char-by-char (`ord(c) % vocab_size`,
  0-padded). `src/mini_networks/models/clip/data.py` re-exports these.
- `src/mini_networks/models/clip/trainer.py` — `CLIPTrainer` subclasses the
  shared `ContrastiveTrainer` runtime; `infer()` returns `image_embeds` or
  `text_embeds` depending on the input key.

## Contrastive cousins

Three single-modality models learn without labels — same spirit, three
different tricks:

- `src/mini_networks/models/simclr/` — `SimCLREncoder` (CNN + projection
  head, normalized output) trained with `info_nce_loss(z1, z2)`: two
  augmented views of the same image are positives; the 2B×2B similarity
  matrix has its diagonal masked with `-1e9` and each view must pick its
  partner out of the batch.
- `src/mini_networks/models/vision_embed/` — `VisionEmbedCNN` with a simpler
  one-directional InfoNCE in `VisionEmbedTrainer._loss()`:
  `cross_entropy((emb_a @ emb_b.T) / temperature, arange(B))`.
- `src/mini_networks/models/dino/` — `MiniDINO`: no negatives at all.
  A student ViT matches the output distribution of an EMA *teacher* of
  itself on the other view (self-distillation); collapse is prevented by
  centering + a sharper teacher temperature. Reuses the supervised `vit`
  backbone via `MiniViT.forward_features`.

All three train on the `task="contrastive"` MNIST mode from the data
registry.

## Multimodal blocks

`src/mini_networks/models/multimodal/` generalizes the dual-encoder idea into
reusable parts:

- `encoders.py` — `VisionCNNEncoder` (pooled vector), `VisionPatchEncoder`
  (ViT-style patch tokens with learned positions), `TextEncoder`,
  `AudioConvEncoder` (1D convs over waveforms → token sequence),
  `TabularFeatureEncoder` (one token per feature), plus `pool_sequence()`.
- `fusion.py` — `ConcatFusion` (concat + project), `GatedFusion` (sigmoid
  gate between two vectors), `CrossAttentionBlock` / `CrossAttentionFusion`
  (query sequence attends to context sequence, then pools).
- `blocks.py` — `MultiModalEncoder` (image + text with pluggable fusion) and
  `CrossModalEncoder` (vision, audio, or tabular paired with text).

These power the contrastive compositions in
`src/mini_networks/compositions/`: `audio_text_dual_encoder.py` and
`tabular_text_dual_encoder.py` are CLIP-style dual encoders over speech
digits and Iris rows, while `tabular_text_cross_attention.py` and
`audio_text_contrastive.py` route one modality through cross-attention with
text before the contrastive loss. All extend `ContrastiveCompositionBase`
in `compositions/base.py`, which owns the training loop and caption
tokenization.

## Try it

```bash
uv run python main.py train --model clip --fast_demo
uv run python main.py train --model simclr --fast_demo
```

## Latest results

<!-- results:start items=clip,simclr,dino,vision_embed -->

_Latest sweep: tier S on cpu_

| Item | Status | Metric | Value | Threshold |
|---|---|---|---|---|
| clip | pass | eval_loss | 1.4359 | n/a |
| simclr | pass | eval_loss | 1.6967 | n/a |
| vision_embed | pass | eval_loss | 1.0148 | n/a |

<!-- results:end -->
