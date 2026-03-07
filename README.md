# mini_networks

An educational ML playground that unifies vision, language, RL, and multimodal experiments into a single, Colab-friendly product. The focus is on consistency: one runtime contract, one data registry, one logging format, and clear compositions that show how models can interoperate.

**Quick Start**
1. Install dependencies.
```
uv sync
```
2. List available models and compositions.
```
python main.py list
```
3. Train a model (fast demo).
```
python main.py train --model diffusion --fast_demo
```
4. Evaluate from a checkpoint.
```
python main.py evaluate --model diffusion --checkpoint runs/diffusion/<timestamp>/artifacts
```
5. Run a composition.
```
python main.py compose --composition clip_guided_diffusion --fast_demo
```

**What’s Inside**
- `src/mini_networks/` contains the unified runtime, data registry, and model implementations.
- `legacy/` is reference-only and will be removed or archived later.
- `runs/` stores all outputs in a consistent structure.

**Refactor Notes**
- Trainer logic is centralized in shared base classes (`SupervisedTrainer`, `ContrastiveTrainer`, etc.).
- Diffusion sampling uses a single driver to keep guidance + logging consistent.
- Multimodal contrastive compositions share a base and expose `infer(config, inputs)` with standard keys.

**Models**
Vision:
- `classifier` Small CNN classifier for MNIST/FashionMNIST.
- `mobilenet` Tiny MobileNet-like CNN baseline.
- `convnext` Tiny ConvNeXt-like CNN baseline.
- `resnet` Mini ResNet baseline.
- `vit` Mini ViT baseline.
- `segmentation` UNet binary or multiclass segmentation.
- `detection` YOLO-style digit localization on 56×56 canvas.
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
- `audio_transformer` Transformer over spectrogram frames.
- `audio_melspectrogram` 2D CNN over mel-spectrograms.

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

**Compositions**
- `clip_guided_diffusion` CLIP similarity guides diffusion sampling.
- `transformer_clip_diffusion` LM → CLIP → diffusion pipeline.
- `gan_diffusion_comparison` side‑by‑side visual comparison.
- `clip_guided_gan` CLIP similarity in GAN training.
- `classifier_guided_diffusion` classifier gradient guidance for diffusion.
- `rag_guided_generation` retrieve context then generate.
- `lora_lm` LoRA adapter fine‑tuning for TransformerLM.
- `segment_then_detect` segmentation → bbox.
- `multitask_vision` shared encoder with multiple heads.
- `diffusion_distillation` teacher/student diffusion distillation.
- `audio_text_contrastive` audio-text contrastive alignment.
- `tabular_text_cross_attention` tabular-text cross-attention alignment.
- `audio_text_dual_encoder` audio-text dual-encoder.
- `tabular_text_dual_encoder` tabular-text dual-encoder.
- `classifier_guided_gan` classifier-guided GAN.
- `rag_conditioned_diffusion` RAG-conditioned diffusion.
- `image_captioning` image captioning from MNIST digits.
- `multimodal_fusion_baseline` image+text fusion baseline.
- `latent_diffusion` latent diffusion (VAE + UNet).

**Data Registry**
The shared registry lives in `src/mini_networks/core/data/registry.py`.
- Vision datasets: `mnist`, `fashion_mnist`.
- Vision task modes: `classification`, `binary_segmentation`, `multiclass_segmentation`, `detection`, `clip`, `contrastive`.
- Text datasets: `text_file`, `tiny_shakespeare`.

**Logging and Outputs**
Every run writes to:
- `runs/<project>/<timestamp>/metrics.jsonl`
- `runs/<project>/<timestamp>/config.yaml`
- `runs/<project>/<timestamp>/artifacts/`

**Colab**
Use the launcher for a guided workflow:
```
from mini_networks.colab.launcher import interactive_menu
interactive_menu()
```

**API Server**
```
python main.py serve --host 0.0.0.0 --port 8000
```
API docs are served at `/docs`.

Composition endpoints:
- `POST /compose/{composition_name}` starts training in the background.
- `POST /compose/{composition_name}/infer` runs inference for a composition.

**Docs**
Start here:
- `docs/README.md`
- `docs/datasets.md`
- `docs/multimodal_blocks.md`
- `docs/models.md`
- `docs/compositions.md`
- `docs/runs_and_logging.md`

**Tests**
```
pytest
```
Smoke tests are in `tests/test_smoke_models.py` and `tests/test_smoke_registry.py`.
