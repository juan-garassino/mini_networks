"""Catalog of runnable items: names, descriptions, display categories."""
from __future__ import annotations

from mini_networks.core.registry import MODEL_NAMES as MODELS  # noqa: F401

COMPOSITIONS = [
    "clip_guided_diffusion",
    "transformer_clip_diffusion",
    "gan_diffusion_comparison",
    "clip_guided_gan",
    "classifier_guided_diffusion",
    "rag_guided_generation",
    "lora_lm",
    "segment_then_detect",
    "multitask_vision",
    "diffusion_distillation",
    "audio_text_contrastive",
    "tabular_text_cross_attention",
    "audio_text_dual_encoder",
    "tabular_text_dual_encoder",
    "classifier_guided_gan",
    "rag_conditioned_diffusion",
    "image_captioning",
    "multimodal_fusion_baseline",
    "latent_diffusion",
]

DESCRIPTIONS = {
    "clip":                          "Contrastive image–text matching on MNIST",
    "diffusion":                     "DDPM denoising with EMA + curriculum learning",
    "segmentation":                  "UNet binary / multiclass segmentation on MNIST",
    "detection":                     "YOLO-style digit localisation on 56×56 canvas",
    "classifier":                    "Small CNN classifier baseline on MNIST/Fashion",
    "resnet":                        "Mini ResNet baseline on MNIST/Fashion",
    "vit":                           "Mini ViT baseline on MNIST/Fashion",
    "vae":                           "Conv VAE reconstruction on MNIST/Fashion",
    "unet_ae":                       "UNet autoencoder reconstruction",
    "simclr":                        "SimCLR-lite contrastive vision pretraining",
    "dino":                          "DINO self-distillation ViT (EMA teacher, no labels)",
    "transformer":                   "Character-level TransformerLM on Shakespeare",
    "mamba":                         "NanoMamba state-space sequence model",
    "gan":                           "Generator + Discriminator trained on MNIST",
    "rnn":                           "RNN / LSTM / GRU recurrent language model",
    "lora":                          "Low-rank fine-tuning: MNIST → FashionMNIST",
    "rag":                           "TF-IDF retrieval + TransformerLM generation",
    "rl_maze":                       "Q / DQN / PPO agents on a procedural maze",
    "rlhf":                          "PPO fine-tuning with Shakespearean reward",
    "reinforce":                     "REINFORCE policy gradient on a procedural maze",
    "audio_classifier":              "1D CNN classifier on speech digits",
    "audio_spectrogram":             "2D CNN on audio spectrograms",
    "audio_transformer":             "Transformer over spectrogram frames",
    "audio_melspectrogram":          "2D CNN on mel-spectrograms",
    "tabular_classifier":            "MLP classifier on Iris (tabular)",
    "tabular_diffusion":             "Diffusion for tabular data synthesis",
    "mobilenet":                     "Tiny MobileNet-like CNN baseline",
    "convnext":                      "Tiny ConvNeXt-like CNN baseline",
    "vision_embed":                  "Vision embedding encoder (contrastive)",
    "text_seq2seq":                  "Transformer encoder-decoder (seq2seq)",
    "text_token_classifier":         "Token classifier (vowel vs other)",
    "pixelcnn":                      "PixelCNN-lite autoregressive model",
    "clip_guided_diffusion":         "CLIP + Diffusion — text-guided image generation",
    "transformer_clip_diffusion":    "Transformer + CLIP + Diffusion — LM steers generation",
    "gan_diffusion_comparison":      "GAN vs Diffusion — side-by-side educational comparison",
    "clip_guided_gan":               "GAN guided by CLIP similarity",
    "classifier_guided_diffusion":   "Classifier-guided diffusion sampling",
    "rag_guided_generation":         "Retrieve context then generate (RAG-guided)",
    "lora_lm":                        "LoRA adapter fine-tuning for TransformerLM",
    "segment_then_detect":           "Segmentation then bbox detection",
    "multitask_vision":              "Shared encoder with cls + seg + det heads",
    "diffusion_distillation":        "Distill diffusion teacher into small denoiser",
    "audio_text_contrastive":        "Audio-text contrastive alignment (speech digits)",
    "tabular_text_cross_attention":  "Tabular-text cross-attention alignment (Iris)",
    "audio_text_dual_encoder":       "Audio-text dual-encoder contrastive",
    "tabular_text_dual_encoder":     "Tabular-text dual-encoder contrastive",
    "classifier_guided_gan":         "Classifier-guided GAN",
    "rag_conditioned_diffusion":     "RAG-conditioned diffusion",
    "image_captioning":              "Image captioning (MNIST)",
    "multimodal_fusion_baseline":    "Image+text fusion classifier",
    "latent_diffusion":              "Latent diffusion (VAE + UNet)",
}

CATEGORY = {name: "Vision / Multimodal" for name in [
    "clip", "diffusion", "segmentation", "detection", "gan",
    "classifier", "resnet", "vit", "vae", "unet_ae", "simclr", "dino",
]}
CATEGORY.update({name: "Language" for name in ["transformer", "mamba", "rnn", "lora", "rag", "rlhf"]})
CATEGORY["rl_maze"] = "Reinforcement Learning"
CATEGORY["reinforce"] = "Reinforcement Learning"
CATEGORY["audio_classifier"] = "Audio"
CATEGORY["audio_spectrogram"] = "Audio"
CATEGORY["audio_transformer"] = "Audio"
CATEGORY["audio_melspectrogram"] = "Audio"
CATEGORY["tabular_classifier"] = "Tabular"
CATEGORY["tabular_diffusion"] = "Tabular"
CATEGORY["mobilenet"] = "Vision / Multimodal"
CATEGORY["convnext"] = "Vision / Multimodal"
CATEGORY["vision_embed"] = "Vision / Multimodal"
CATEGORY["text_seq2seq"] = "Language"
CATEGORY["text_token_classifier"] = "Language"
CATEGORY["pixelcnn"] = "Vision / Multimodal"
CATEGORY["clip_guided_diffusion"] = "Composition"
CATEGORY["transformer_clip_diffusion"] = "Composition"
CATEGORY["gan_diffusion_comparison"] = "Composition"
CATEGORY["clip_guided_gan"] = "Composition"
CATEGORY["classifier_guided_diffusion"] = "Composition"
CATEGORY["rag_guided_generation"] = "Composition"
CATEGORY["lora_lm"] = "Composition"
CATEGORY["segment_then_detect"] = "Composition"
CATEGORY["multitask_vision"] = "Composition"
CATEGORY["diffusion_distillation"] = "Composition"
CATEGORY["audio_text_contrastive"] = "Composition"
CATEGORY["tabular_text_cross_attention"] = "Composition"
CATEGORY["audio_text_dual_encoder"] = "Composition"
CATEGORY["tabular_text_dual_encoder"] = "Composition"
CATEGORY["classifier_guided_gan"] = "Composition"
CATEGORY["rag_conditioned_diffusion"] = "Composition"
CATEGORY["image_captioning"] = "Composition"
CATEGORY["multimodal_fusion_baseline"] = "Composition"
CATEGORY["latent_diffusion"] = "Composition"
