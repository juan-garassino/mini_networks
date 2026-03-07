# Models Overview

Short descriptions of all model families in the unified runtime.

**Original Experiments (Legacy → Unified)**
- `clip` Contrastive image–text matching on MNIST.
- `diffusion` DDPM denoising baseline.
- `segmentation` UNet for segmentation.
- `detection` YOLO-style digit localization.
- `gan` GAN baseline for MNIST.
- `transformer` Character-level TransformerLM.
- `mamba` NanoMamba state-space model.
- `rnn` RNN/LSTM/GRU language model.
- `lora` LoRA fine‑tuning (MNIST → FashionMNIST).
- `rag` TF‑IDF retrieval + TransformerLM.
- `rlhf` PPO fine‑tuning with heuristic reward.
- `rl_maze` Q / DQN / PPO maze agents.

**Full Inventory (All Unified Models)**
Vision:
- `classifier` Small CNN classifier for MNIST/FashionMNIST.
- `mobilenet` Tiny MobileNet-like CNN baseline.
- `convnext` Tiny ConvNeXt-like CNN baseline.
- `resnet` Mini ResNet baseline.
- `vit` Mini ViT baseline.
- `segmentation` UNet for binary/multiclass segmentation.
- `detection` YOLO-style digit localization.
- `vision_embed` Contrastive vision embedding encoder.
- `simclr` SimCLR-lite self-supervised pretraining.

Generative vision:
- `gan` MLP GAN baseline for MNIST.
- `diffusion` DDPM denoising with EMA + curriculum options.
- `pixelcnn` PixelCNN-lite autoregressive model.
- `vae` Convolutional VAE reconstruction.
- `unet_ae` UNet autoencoder reconstruction.

Audio:
- `audio_classifier` 1D CNN classifier on speech digits.
- `audio_spectrogram` 2D CNN over STFT magnitude.
- `audio_melspectrogram` 2D CNN over mel-spectrograms.
- `audio_transformer` Transformer over spectrogram frames.

Tabular:
- `tabular_classifier` MLP/linear/transformer classifier (Iris).
- `tabular_diffusion` Diffusion model for tabular synthesis.

Language:
- `transformer` Character-level TransformerLM on Tiny Shakespeare.
- `mamba` NanoMamba state-space model.
- `rnn` RNN / LSTM / GRU language model.
- `text_seq2seq` Transformer encoder-decoder.
- `text_token_classifier` Token classifier (vowel vs other).
- `rag` TF‑IDF retrieval + TransformerLM generation.
- `rlhf` PPO fine‑tuning with heuristic Shakespearean reward.
- `lora` LoRA adapter fine‑tuning.

Multimodal:
- `clip` Contrastive image–text matching on MNIST.

Reinforcement learning:
- `rl_maze` Q / DQN / PPO agents on a procedural maze.
- `reinforce` REINFORCE policy gradient on a procedural maze.
