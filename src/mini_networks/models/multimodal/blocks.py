"""Multimodal encoder wrappers: pair text with a second modality, fuse, get one vector.

Key idea: a multimodal model is three decisions — how to encode each modality,
whether to fuse at the level of pooled vectors (late fusion) or token sequences
(cross-attention), and how to pool. These wrappers make each decision a config
string so the trade-offs can be compared on the same data.

This implementation (d_model=128 throughout): MultiModalEncoder pairs a
TextEncoder (token + pos embedding, 2-layer Transformer encoder) with MNIST-style
images, encoded two ways — VisionPatchEncoder slices 28x28 into 4x4 patches via
a strided Conv2d, giving 49 tokens of 128 dims for cross-attention, while
VisionCNNEncoder produces a single pooled 128-dim vector for the late-fusion
paths. The fusion flag picks ConcatFusion (project [text; image] of 256 → 128),
GatedFusion (g = sigmoid(W[a; b]); out = g*a + (1-g)*b, a learned convex blend),
or CrossAttentionFusion (text tokens attend to image tokens, then mean/cls pool).
CrossModalEncoder generalises the second branch to vision, audio (Conv1d frame
tokens), or tabular (one token per feature) against the same text encoder.

Deliberately simplified vs CLIP/Flamingo/ViLBERT-class models: a single fusion
block rather than interleaved cross-attention layers, no contrastive or any other
pretraining objective baked in, no modality-type embeddings, and attention runs
in one direction only (text queries the other modality, never the reverse).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mini_networks.models.multimodal.encoders import (
    VisionCNNEncoder,
    VisionPatchEncoder,
    TextEncoder,
    AudioConvEncoder,
    TabularFeatureEncoder,
    pool_sequence,
)
from mini_networks.models.multimodal.fusion import (
    ConcatFusion,
    GatedFusion,
    CrossAttentionFusion,
)


class MultiModalEncoder(nn.Module):
    """Simple multimodal encoder with pluggable fusion."""

    def __init__(
        self,
        d_model: int = 128,
        vocab_size: int = 256,
        fusion: str = "cross_attention",
        pool: str = "mean",
    ):
        super().__init__()
        self.text = TextEncoder(vocab_size=vocab_size, d_model=d_model)
        self.vision_tokens = VisionPatchEncoder(d_model=d_model)
        self.vision_pool = VisionCNNEncoder(out_dim=d_model)
        self.pool = pool

        if fusion == "concat":
            self.fusion = ConcatFusion(d_model, d_model)
        elif fusion == "gated":
            self.fusion = GatedFusion(d_model)
        elif fusion == "cross_attention":
            self.fusion = CrossAttentionFusion(d_model=d_model, pool=pool)
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.text(tokens)

    def encode_image_tokens(self, images: torch.Tensor) -> torch.Tensor:
        return self.vision_tokens(images)

    def encode_image_pooled(self, images: torch.Tensor) -> torch.Tensor:
        return self.vision_pool(images)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        text_tokens = self.encode_text(tokens)
        image_tokens = self.encode_image_tokens(images)

        if isinstance(self.fusion, CrossAttentionFusion):
            return self.fusion(text_tokens, image_tokens)

        text_vec = pool_sequence(text_tokens, mode=self.pool)
        image_vec = self.encode_image_pooled(images)
        return self.fusion(text_vec, image_vec)


class CrossModalEncoder(nn.Module):
    """Cross-modal encoder that supports vision, audio, or tabular with text."""

    def __init__(
        self,
        modality: str = "vision",
        d_model: int = 128,
        vocab_size: int = 256,
        fusion: str = "cross_attention",
        pool: str = "mean",
        n_features: int = 8,
    ):
        super().__init__()
        self.modality = modality
        self.text = TextEncoder(vocab_size=vocab_size, d_model=d_model)
        self.pool = pool

        if modality == "vision":
            self.mod_tokens = VisionPatchEncoder(d_model=d_model)
            self.mod_pool = VisionCNNEncoder(out_dim=d_model)
        elif modality == "audio":
            self.mod_tokens = AudioConvEncoder(d_model=d_model)
            self.mod_pool = None
        elif modality == "tabular":
            self.mod_tokens = TabularFeatureEncoder(n_features=n_features, d_model=d_model)
            self.mod_pool = None
        else:
            raise ValueError(f"Unknown modality: {modality}")

        if fusion == "concat":
            self.fusion = ConcatFusion(d_model, d_model)
        elif fusion == "gated":
            self.fusion = GatedFusion(d_model)
        elif fusion == "cross_attention":
            self.fusion = CrossAttentionFusion(d_model=d_model, pool=pool)
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

    def forward(self, modality_inputs: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        text_tokens = self.text(tokens)
        mod_tokens = self.mod_tokens(modality_inputs)

        if isinstance(self.fusion, CrossAttentionFusion):
            return self.fusion(text_tokens, mod_tokens)

        text_vec = pool_sequence(text_tokens, mode=self.pool)
        if self.mod_pool is not None:
            mod_vec = self.mod_pool(modality_inputs)
        else:
            mod_vec = pool_sequence(mod_tokens, mode="mean")
        return self.fusion(text_vec, mod_vec)
