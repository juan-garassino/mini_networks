"""Smoke tests for multimodal building blocks."""
import torch

from mini_networks.models.multimodal.blocks import MultiModalEncoder, CrossModalEncoder
from mini_networks.models.multimodal.fusion import CrossAttentionBlock


def test_cross_attention_block_shapes():
    block = CrossAttentionBlock(d_model=32, n_heads=4)
    q = torch.randn(2, 5, 32)
    ctx = torch.randn(2, 7, 32)
    out = block(q, ctx)
    assert out.shape == q.shape


def test_multimodal_encoder_forward():
    model = MultiModalEncoder(d_model=32, fusion="cross_attention")
    images = torch.randn(2, 1, 28, 28)
    tokens = torch.randint(0, 256, (2, 16))
    out = model(images, tokens)
    assert out.shape == (2, 32)


def test_crossmodal_audio_forward():
    model = CrossModalEncoder(modality="audio", d_model=32, fusion="cross_attention")
    wave = torch.randn(2, 1, 256)
    tokens = torch.randint(0, 256, (2, 16))
    out = model(wave, tokens)
    assert out.shape == (2, 32)


def test_crossmodal_tabular_forward():
    model = CrossModalEncoder(modality="tabular", d_model=32, fusion="cross_attention", n_features=8)
    x = torch.randn(2, 8)
    tokens = torch.randint(0, 256, (2, 16))
    out = model(x, tokens)
    assert out.shape == (2, 32)
