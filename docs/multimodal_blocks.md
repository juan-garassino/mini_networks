# Multimodal Building Blocks

This repo exposes lightweight, composable components for cross‑modal experiments in:
`src/mini_networks/models/multimodal/`

**Encoders**
- `VisionCNNEncoder` pooled image embedding.
- `VisionPatchEncoder` image → patch tokens.
- `TextEncoder` token embeddings + Transformer encoder.
- `AudioConvEncoder` waveform → token sequence.
- `TabularFeatureEncoder` features → token sequence.

**Fusion**
- `ConcatFusion` concat pooled vectors.
- `GatedFusion` learn a soft gate between pooled vectors.
- `CrossAttentionBlock` generic cross‑attention (query attends to context).
- `CrossAttentionFusion` cross‑attention + pooling.

**Wrapper**
- `MultiModalEncoder` combines text + vision with a selectable fusion strategy.
- `CrossModalEncoder` combines text with vision, audio, or tabular.

**Usage sketch**
```
from mini_networks.models.multimodal.blocks import MultiModalEncoder
model = MultiModalEncoder(d_model=128, fusion="cross_attention")
emb = model(images, tokens)
```

Audio/text:
```
from mini_networks.models.multimodal.blocks import CrossModalEncoder
model = CrossModalEncoder(modality="audio", d_model=128, fusion="cross_attention")
emb = model(waveforms, tokens)
```

These blocks are designed to be simple and reusable in compositions (CLIP‑guided GAN, LM‑guided diffusion, etc.).
