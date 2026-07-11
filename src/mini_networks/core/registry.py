"""Model registry: the single source of truth for what models exist.

`MODEL_NAMES` is a static list so CLIs and catalogs can enumerate models
without importing torch; `get_model_registry()` does the heavy imports
lazily and is cached. A unit test asserts the two stay in sync.
"""
from __future__ import annotations

from functools import lru_cache

MODEL_NAMES: list[str] = [
    "clip",
    "diffusion",
    "segmentation",
    "detection",
    "classifier",
    "resnet",
    "vit",
    "vae",
    "unet_ae",
    "simclr",
    "dino",
    "transformer",
    "mamba",
    "gan",
    "rnn",
    "lora",
    "rag",
    "rl_maze",
    "rlhf",
    "grpo",
    "reinforce",
    "audio_classifier",
    "audio_spectrogram",
    "audio_transformer",
    "audio_melspectrogram",
    "tabular_classifier",
    "mobilenet",
    "convnext",
    "vision_embed",
    "text_seq2seq",
    "text_token_classifier",
    "pixelcnn",
    "tabular_diffusion",
]


@lru_cache(maxsize=1)
def get_model_registry() -> dict:
    """name → (ConfigClass, TrainerClass, dataloader_fn)."""
    from mini_networks.models.clip.config import CLIPConfig
    from mini_networks.models.clip.trainer import CLIPTrainer, make_clip_dataloader
    from mini_networks.models.diffusion.config import DiffusionConfig
    from mini_networks.models.diffusion.trainer import DDPMTrainer, make_diffusion_dataloader
    from mini_networks.models.segmentation.config import SegmentationConfig
    from mini_networks.models.segmentation.trainer import SegmentationTrainer, make_segmentation_dataloader
    from mini_networks.models.detection.config import DetectionConfig
    from mini_networks.models.detection.trainer import DetectionTrainer, make_detection_dataloader
    from mini_networks.models.transformer.config import TransformerConfig
    from mini_networks.models.transformer.trainer import TransformerTrainer, make_transformer_dataloader
    from mini_networks.models.mamba.config import MambaConfig
    from mini_networks.models.mamba.trainer import MambaTrainer, make_mamba_dataloader
    from mini_networks.models.gan.config import GANConfig
    from mini_networks.models.gan.trainer import GANTrainer, make_gan_dataloader
    from mini_networks.models.rnn.config import RNNConfig
    from mini_networks.models.rnn.trainer import RNNTrainer, make_rnn_dataloader
    from mini_networks.models.lora.config import LoRAConfig
    from mini_networks.models.lora.trainer import LoRATrainer, make_lora_dataloader
    from mini_networks.models.rag.config import RAGConfig
    from mini_networks.models.rag.trainer import RAGTrainer, make_rag_dataloader
    from mini_networks.models.rl_maze.config import RLMazeConfig
    from mini_networks.models.rl_maze.trainer import RLMazeTrainer, make_rl_maze_dataloader
    from mini_networks.models.rlhf.config import RLHFConfig
    from mini_networks.models.rlhf.trainer import RLHFTrainer, make_rlhf_dataloader
    from mini_networks.models.grpo.config import GRPOConfig
    from mini_networks.models.grpo.trainer import GRPOTrainer, make_grpo_dataloader
    from mini_networks.models.vae.config import VAEConfig
    from mini_networks.models.vae.trainer import VAETrainer, make_vae_dataloader
    from mini_networks.models.classifier.config import ClassifierConfig
    from mini_networks.models.classifier.trainer import ClassifierTrainer, make_classifier_dataloader
    from mini_networks.models.resnet.config import ResNetConfig
    from mini_networks.models.resnet.trainer import ResNetTrainer, make_resnet_dataloader
    from mini_networks.models.vit.config import ViTConfig
    from mini_networks.models.vit.trainer import ViTTrainer, make_vit_dataloader
    from mini_networks.models.simclr.config import SimCLRConfig
    from mini_networks.models.simclr.trainer import SimCLRTrainer, make_simclr_dataloader
    from mini_networks.models.dino.config import DINOConfig
    from mini_networks.models.dino.trainer import DINOTrainer, make_dino_dataloader
    from mini_networks.models.unet_ae.config import UNetAEConfig
    from mini_networks.models.unet_ae.trainer import UNetAETrainer, make_unet_ae_dataloader
    from mini_networks.models.reinforce.config import ReinforceConfig
    from mini_networks.models.reinforce.trainer import ReinforceTrainer, make_reinforce_dataloader
    from mini_networks.models.audio.config import (
        AudioClassifierConfig,
        AudioSpecClassifierConfig,
        AudioMelSpecClassifierConfig,
        AudioTransformerConfig,
    )
    from mini_networks.models.audio.trainer import (
        AudioClassifierTrainer,
        make_audio_dataloader,
        AudioSpecClassifierTrainer,
        make_audio_spec_dataloader,
        AudioTransformerTrainer,
        make_audio_transformer_dataloader,
        AudioMelSpecClassifierTrainer,
        make_audio_melspec_dataloader,
    )
    from mini_networks.models.tabular.config import TabularClassifierConfig
    from mini_networks.models.tabular.trainer import TabularClassifierTrainer, make_tabular_dataloader
    from mini_networks.models.mobilenet.config import MobileNetConfig
    from mini_networks.models.mobilenet.trainer import MobileNetTrainer, make_mobilenet_dataloader
    from mini_networks.models.vision_embed.config import VisionEmbedConfig
    from mini_networks.models.vision_embed.trainer import VisionEmbedTrainer, make_vision_embed_dataloader
    from mini_networks.models.text_seq2seq.config import TextSeq2SeqConfig
    from mini_networks.models.text_seq2seq.trainer import TextSeq2SeqTrainer, make_text_seq2seq_dataloader
    from mini_networks.models.text_token_classifier.config import TextTokenClassifierConfig
    from mini_networks.models.text_token_classifier.trainer import (
        TextTokenClassifierTrainer,
        make_text_token_classifier_dataloader,
    )
    from mini_networks.models.pixelcnn.config import PixelCNNConfig
    from mini_networks.models.pixelcnn.trainer import PixelCNNTrainer, make_pixelcnn_dataloader
    from mini_networks.models.tabular_diffusion.config import TabularDiffusionConfig
    from mini_networks.models.tabular_diffusion.trainer import (
        TabularDiffusionTrainer,
        make_tabular_diffusion_dataloader,
    )
    from mini_networks.models.convnext.config import ConvNeXtConfig
    from mini_networks.models.convnext.trainer import ConvNeXtTrainer, make_convnext_dataloader

    return {
        "clip": (CLIPConfig, CLIPTrainer, make_clip_dataloader),
        "diffusion": (DiffusionConfig, DDPMTrainer, make_diffusion_dataloader),
        "segmentation": (SegmentationConfig, SegmentationTrainer, make_segmentation_dataloader),
        "detection": (DetectionConfig, DetectionTrainer, make_detection_dataloader),
        "classifier": (ClassifierConfig, ClassifierTrainer, make_classifier_dataloader),
        "resnet": (ResNetConfig, ResNetTrainer, make_resnet_dataloader),
        "vit": (ViTConfig, ViTTrainer, make_vit_dataloader),
        "vae": (VAEConfig, VAETrainer, make_vae_dataloader),
        "unet_ae": (UNetAEConfig, UNetAETrainer, make_unet_ae_dataloader),
        "simclr": (SimCLRConfig, SimCLRTrainer, make_simclr_dataloader),
        "dino": (DINOConfig, DINOTrainer, make_dino_dataloader),
        "transformer": (TransformerConfig, TransformerTrainer, make_transformer_dataloader),
        "mamba": (MambaConfig, MambaTrainer, make_mamba_dataloader),
        "gan": (GANConfig, GANTrainer, make_gan_dataloader),
        "rnn": (RNNConfig, RNNTrainer, make_rnn_dataloader),
        "lora": (LoRAConfig, LoRATrainer, make_lora_dataloader),
        "rag": (RAGConfig, RAGTrainer, make_rag_dataloader),
        "rl_maze": (RLMazeConfig, RLMazeTrainer, make_rl_maze_dataloader),
        "rlhf": (RLHFConfig, RLHFTrainer, make_rlhf_dataloader),
        "grpo": (GRPOConfig, GRPOTrainer, make_grpo_dataloader),
        "reinforce": (ReinforceConfig, ReinforceTrainer, make_reinforce_dataloader),
        "audio_classifier": (AudioClassifierConfig, AudioClassifierTrainer, make_audio_dataloader),
        "audio_spectrogram": (AudioSpecClassifierConfig, AudioSpecClassifierTrainer, make_audio_spec_dataloader),
        "audio_transformer": (AudioTransformerConfig, AudioTransformerTrainer, make_audio_transformer_dataloader),
        "audio_melspectrogram": (AudioMelSpecClassifierConfig, AudioMelSpecClassifierTrainer, make_audio_melspec_dataloader),
        "tabular_classifier": (TabularClassifierConfig, TabularClassifierTrainer, make_tabular_dataloader),
        "mobilenet": (MobileNetConfig, MobileNetTrainer, make_mobilenet_dataloader),
        "convnext": (ConvNeXtConfig, ConvNeXtTrainer, make_convnext_dataloader),
        "vision_embed": (VisionEmbedConfig, VisionEmbedTrainer, make_vision_embed_dataloader),
        "text_seq2seq": (TextSeq2SeqConfig, TextSeq2SeqTrainer, make_text_seq2seq_dataloader),
        "text_token_classifier": (TextTokenClassifierConfig, TextTokenClassifierTrainer, make_text_token_classifier_dataloader),
        "pixelcnn": (PixelCNNConfig, PixelCNNTrainer, make_pixelcnn_dataloader),
        "tabular_diffusion": (TabularDiffusionConfig, TabularDiffusionTrainer, make_tabular_diffusion_dataloader),
    }
