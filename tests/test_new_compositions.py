"""Smoke tests for newly added compositions."""
import os
import tempfile

import torch

from mini_networks.core.logging.logger import Logger
from mini_networks.compositions.classifier_guided_diffusion import (
    ClassifierGuidedDiffusion,
    ClassifierGuidedDiffusionConfig,
)
from mini_networks.compositions.clip_guided_gan import CLIPGuidedGAN, CLIPGuidedGANConfig
from mini_networks.compositions.rag_guided_generation import (
    RAGGuidedGeneration,
    RAGGuidedGenerationConfig,
)
from mini_networks.compositions.lora_lm import LoRALM, LoRALMConfig
from mini_networks.compositions.segment_then_detect import SegmentThenDetect, SegmentThenDetectConfig
from mini_networks.compositions.multitask_vision import MultiTaskVision, MultiTaskVisionConfig
from mini_networks.compositions.diffusion_distillation import (
    DiffusionDistillation,
    DiffusionDistillationConfig,
)

DATA_ROOT = os.environ.get("MINI_TEST_DATA_ROOT", "/tmp/mini_networks_test_data")


def test_classifier_guided_diffusion_smoke():
    cfg = ClassifierGuidedDiffusionConfig(fast_demo=True, data_root=DATA_ROOT, timesteps=20)
    comp = ClassifierGuidedDiffusion()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir, "test")
        comp.run(cfg, logger)
        samples = comp.sample(cfg, n=2)
        assert samples.shape == (2, 1, 28, 28)


def test_clip_guided_gan_smoke():
    cfg = CLIPGuidedGANConfig(fast_demo=True, data_root=DATA_ROOT)
    comp = CLIPGuidedGAN()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir, "test")
        comp.train(cfg, logger)
        out = comp.sample(cfg, n=2)
        assert out.shape == (2, 1, 28, 28)


def test_rag_guided_generation_smoke():
    cfg = RAGGuidedGenerationConfig(fast_demo=True, data_root=DATA_ROOT)
    comp = RAGGuidedGeneration()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir, "test")
        comp.train(cfg, logger)
        text = comp.generate(cfg, "To be or not to be", max_new_tokens=8)
        assert isinstance(text, str)


def test_lora_lm_smoke():
    cfg = LoRALMConfig(fast_demo=True, data_root=DATA_ROOT)
    comp = LoRALM()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir, "test")
        comp.train(cfg, logger)
        text = comp.generate(cfg, "Hello", max_new_tokens=4)
        assert isinstance(text, str)


def test_segment_then_detect_smoke():
    cfg = SegmentThenDetectConfig(fast_demo=True, data_root=DATA_ROOT)
    comp = SegmentThenDetect()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir, "test")
        comp.train(cfg, logger)
        images = torch.randn(2, 1, 28, 28)
        bboxes = comp.infer_bbox(cfg, images)
        assert bboxes.shape == (2, 4)


def test_multitask_vision_smoke():
    cfg = MultiTaskVisionConfig(fast_demo=True, data_root=DATA_ROOT)
    comp = MultiTaskVision()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir, "test")
        comp.train(cfg, logger)


def test_diffusion_distillation_smoke():
    cfg = DiffusionDistillationConfig(fast_demo=True, data_root=DATA_ROOT, timesteps=20)
    comp = DiffusionDistillation()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir, "test")
        comp.train(cfg, logger)


def test_audio_text_contrastive_smoke():
    from mini_networks.compositions.audio_text_contrastive import (
        AudioTextContrastive,
        AudioTextContrastiveConfig,
    )
    cfg = AudioTextContrastiveConfig(fast_demo=True, data_root=DATA_ROOT, require_downloads=False)
    comp = AudioTextContrastive()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir, "test")
        comp.train(cfg, logger)


def test_tabular_text_cross_attention_smoke():
    from mini_networks.compositions.tabular_text_cross_attention import (
        TabularTextCrossAttention,
        TabularTextCrossAttentionConfig,
    )
    cfg = TabularTextCrossAttentionConfig(fast_demo=True, data_root=DATA_ROOT, require_downloads=False)
    comp = TabularTextCrossAttention()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir, "test")
        comp.train(cfg, logger)


def test_audio_text_dual_encoder_smoke():
    from mini_networks.compositions.audio_text_dual_encoder import (
        AudioTextDualEncoder,
        AudioTextDualEncoderConfig,
    )
    cfg = AudioTextDualEncoderConfig(fast_demo=True, data_root=DATA_ROOT, require_downloads=False)
    comp = AudioTextDualEncoder()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir, "test")
        comp.train(cfg, logger)


def test_tabular_text_dual_encoder_smoke():
    from mini_networks.compositions.tabular_text_dual_encoder import (
        TabularTextDualEncoder,
        TabularTextDualEncoderConfig,
    )
    cfg = TabularTextDualEncoderConfig(fast_demo=True, data_root=DATA_ROOT, require_downloads=False)
    comp = TabularTextDualEncoder()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir, "test")
        comp.train(cfg, logger)


def test_classifier_guided_gan_smoke():
    from mini_networks.compositions.classifier_guided_gan import (
        ClassifierGuidedGAN,
        ClassifierGuidedGANConfig,
    )
    cfg = ClassifierGuidedGANConfig(fast_demo=True, data_root=DATA_ROOT)
    comp = ClassifierGuidedGAN()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir, "test")
        comp.train(cfg, logger)


def test_rag_conditioned_diffusion_smoke():
    from mini_networks.compositions.rag_conditioned_diffusion import (
        RAGConditionedDiffusion,
        RAGConditionedDiffusionConfig,
    )
    cfg = RAGConditionedDiffusionConfig(fast_demo=True, data_root=DATA_ROOT)
    comp = RAGConditionedDiffusion()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir, "test")
        comp.train(cfg, logger)


def test_image_captioning_smoke():
    from mini_networks.compositions.image_captioning import ImageCaptioning, ImageCaptioningConfig
    cfg = ImageCaptioningConfig(fast_demo=True, data_root=DATA_ROOT)
    comp = ImageCaptioning()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir, "test")
        comp.train(cfg, logger)


def test_multimodal_fusion_baseline_smoke():
    from mini_networks.compositions.multimodal_fusion_baseline import (
        MultimodalFusionBaseline,
        MultimodalFusionConfig,
    )
    cfg = MultimodalFusionConfig(fast_demo=True, data_root=DATA_ROOT)
    comp = MultimodalFusionBaseline()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir, "test")
        comp.train(cfg, logger)


def test_latent_diffusion_smoke():
    from mini_networks.compositions.latent_diffusion import LatentDiffusion, LatentDiffusionConfig
    cfg = LatentDiffusionConfig(fast_demo=True, data_root=DATA_ROOT)
    comp = LatentDiffusion()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir, "test")
        comp.train(cfg, logger)
