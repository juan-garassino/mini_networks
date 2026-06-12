# Documentation Index

The docs are an ordered curriculum. Each chapter pairs a short theory recap with
the actual implementation in `src/mini_networks/` and ends with a results table
fed from the latest quality-gate sweep (`uv run python scripts/render_results.py`).

| # | Chapter | Covers |
|---|---|---|
| 00 | [Overview](00-overview.md) | repo map, runtime contract, tiers, quality gate, how to run anything |
| 01 | [Data](01-data.md) | dataset registry, MNIST task modes, text/audio/tabular sets |
| 02 | [Classifiers](02-classifiers.md) | CNN, ResNet, ViT, MobileNet, ConvNeXt |
| 03 | [Autoencoders](03-autoencoders.md) | UNet AE, VAE, latent spaces |
| 04 | [GAN](04-gan.md) | minimax training, mode collapse, judge scoring |
| 05 | [Diffusion](05-diffusion.md) | DDPM, EMA, CFG, tier-capped timesteps, variants |
| 06 | [Sequence models](06-sequence-models.md) | RNN/LSTM/GRU, Transformer (+MoE/Mamba FFN), NanoMamba |
| 07 | [LoRA fine-tuning](07-lora-finetuning.md) | low-rank adapters, two-stage transfer |
| 08 | [RAG](08-rag.md) | TF-IDF retrieval + our own TransformerLM |
| 09 | [CLIP & multimodal](09-clip-multimodal.md) | contrastive dual encoders, fusion blocks |
| 10 | [RL & RLHF](10-rl-rlhf.md) | Q/DQN/PPO maze agents, REINFORCE, PPO-with-KL |
| 11 | [Compositions](11-compositions.md) | multi-model pipelines and the runner contract |

How-to guides:
- [Adding a dataset](adding_datasets.md)

The model source files themselves are annotated: every `models/<name>/model.py`
opens with a header docstring covering the key idea, this implementation's
dimensions, the governing equations, and what is deliberately simplified
versus the original paper.
