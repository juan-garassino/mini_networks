# Project Development Guide

This document explains how to turn the `mini_networks` repository into one cohesive, Colab‑ready educational product. It defines the technical map, the unification plan, and the specific model interconnections (CLIP + Diffusion + Transformers, etc.). Keep this as the source of truth for productization.

## 1. Repository Map (What Exists Today)

**Top level:**
- `legacy/` holds multiple independent experiments.

**Data:**
- `legacy/001-data/` contains raw MNIST and FashionMNIST datasets (binary idx files).

**Experiment families (current state):**
- **Adversarial/GAN:** `legacy/002-adversarial/` (miniGan package, Makefile, scripts).
- **CLIP-style:** `legacy/003-clip/` (miniVisuaLinguist package).
- **Diffusion (TF):** `legacy/004-diffusion/` (miniDiffusion package).
- **Guided Diffusion (Torch):** `legacy/005-guided-diffusion/` (miniGuideDiff package).
- **LoRA/CNN:** `legacy/006-lora/` (CNN + LoRA variants, logging, plotting).
- **RL (Maze):** `legacy/007-maze-rl/` (Q/DQN/PPO, gif generation).
- **Segmentation:** `legacy/008-segmentation/` (placeholder package, no real pipeline yet).
- **Sequence Memory / RNN:** `legacy/009-sequence-memory/` (miniSequenceRememberer / miniRecurrent).
- **Transformer (full training):** `legacy/010-transformer/` (tokenizers, training, generation, Colab mention).
- **Transformer MoE (single file):** `legacy/011-transformer-moe/`.
- **Transformer RLHF/PPO (single file):** `legacy/012-transformer-gptrl/`.
- **Autoregressive Diffusion (single file):** `legacy/013-autoregressive-diff/`.
- **RAG (single file):** `legacy/014-rag/`.

## 2. Product Goal

Build **one educational product** that:
- Runs dynamically in **Colab**.
- Shares **one data loading system**.
- Shares **one logging/artifacts system**.
- Lets users choose a model family (CNN, GAN, Diffusion, Transformer, RL, RAG).
- Supports **cross-model composition** (CLIP‑guided diffusion, Transformer + MoE, etc.).

## 3. Unification Strategy (Step‑by‑Step)

1. **Inventory entrypoints**
   - For every `legacy/<nnn>-<name>/`, identify the primary execution entry (`main.py`, `-run` script, or single-file prototype).

2. **Define a shared runtime contract**
   - Standardize on:
     - `train(config, data_module, logger, output_dir)`
     - `evaluate(config, data_module, logger, output_dir)`

3. **Create a unified Config schema**
   - One config (YAML/JSON + argparse) with shared fields and per‑project overrides.
   - Map legacy flags and environment variables into this schema.

4. **Standardize data loading**
   - Add a dataset registry to `core/data/`.
   - Support: MNIST, FashionMNIST, text corpora, image folders, maze env, synthetic data.
   - Migrate or wrap existing loaders (e.g., `legacy/006-lora/src/data.py`).

5. **Standardize logging**
   - Create one logger that writes to:
     - `runs/<project>/<timestamp>/metrics.jsonl`
     - `runs/<project>/<timestamp>/config.yaml`
     - `runs/<project>/<timestamp>/artifacts/`
   - All plots, checkpoints, and gifs go to `artifacts/`.

6. **Adapter layer per project**
   - Each legacy project gets an adapter mapping its internal code to the shared runtime contract.
   - Do **not** rewrite core model internals; only wrap them.

7. **Composition layer**
   - Define a minimal interface so models can plug together:
     - `encode(inputs)`
     - `score(a, b)`
     - `guided_step(latent, t, guidance_fn)`
     - `sample(config)`
   - Composition adapters stitch multiple models together.

8. **Cross‑model integrations**
   - **CLIP‑guided diffusion**: CLIP provides similarity score; diffusion uses it as guidance.
   - **Diffusion family bridge**: `legacy/004`, `legacy/005`, `legacy/013` share one sampler/driver.
   - **Transformer + MoE**: shared tokenizer + embedding, swap FFN blocks with MoE experts.

9. **Colab entrypoint**
   - Single script `colab_launcher.py` to:
     - Install deps
     - Select model or composition
     - Run training/eval

10. **Validation pass**
   - Smoke test each adapter (1 epoch, small batch, minimal dataset) to confirm data/logging compliance.

## 4. Interconnection Design (How Models Compose)

### 4.1 CLIP + Diffusion
- **Goal:** Text‑guided image generation.
- **Contract:**
  - CLIP adapter exposes `encode(text)` and `encode(image)` + `score(text, image)`.
  - Diffusion adapter exposes `guided_step(latent, t, guidance_fn)`.
- **Mechanism:** Use CLIP similarity to compute gradients that guide diffusion sampling.

### 4.2 Transformer + Diffusion + CLIP
- **Goal:** Prompt generation → guided image generation.
- **Flow:**
  1. Transformer generates prompt variants.
  2. CLIP ranks prompt candidates against desired image embedding.
  3. Diffusion uses best prompt with CLIP guidance.

### 4.3 Diffusion Family Bridge
- **Goal:** Make multiple diffusion variants interchangeable.
- **Mechanism:** Standard sampler API + logging pipeline for:
  - `legacy/004-diffusion` (TensorFlow DDPM)
  - `legacy/005-guided-diffusion` (Torch DDPM)
  - `legacy/013-autoregressive-diff` (Torch hybrid)

### 4.4 Transformer + MoE
- **Goal:** One training loop, multiple FFN backends.
- **Mechanism:**
  - Common tokenizer + embeddings.
  - FFN block is configurable (classic vs MoE).

### 4.5 RAG + Transformer
- **Goal:** Retrieval‑augmented generation.
- **Mechanism:**
  - RAG adapter retrieves context.
  - Transformer adapter conditions generation on retrieved text.

### 4.6 RL + Transformer
- **Goal:** RL‑based fine‑tuning of language models.
- **Mechanism:**
  - Reward model from RLHF prototype (legacy/012).
  - PPO update loop applied to Transformer outputs.

### 4.7 GAN + Diffusion / CNN
- **Goal:** Educational comparison and hybrid pipelines.
- **Mechanism:**
  - Shared dataset + logging.
  - GAN used as baseline or discriminator for diffusion experiments.

### 4.8 CNN/Dense + CLIP
- **Goal:** Simple vision encoders for CLIP‑style training.
- **Mechanism:**
  - Reuse CNN backbone for image embedding.
  - Pair with lightweight text encoder.

## 5. Proposed Product Architecture (Target Layout)

```
core/
  config.py
  runtime.py
  data/
    registry.py
    transforms.py
  logging/
    logger.py
adapters/
  clip/
  diffusion/
  guided_diffusion/
  transformer/
  transformer_moe/
  lora_cnn/
  gan/
  rnn/
  rl_maze/
  rag/
compositions/
  clip_guided_diffusion.py
  transformer_moe.py
  transformer_diffusion_clip.py
  rag_transformer.py
product/
  run.py
colab/
  colab_launcher.py
runs/
```

## 6. Standardized Logging Rules
- All outputs must go under `runs/<project>/<timestamp>/`.
- Required files:
  - `metrics.jsonl`
  - `config.yaml`
- Optional artifacts:
  - `checkpoints/`
  - `plots/`
  - `gifs/`
  - `samples/`

## 7. Standardized Data Rules
- All dataset loaders live in `core/data/registry.py`.
- Use `data/` under each project only as a cache or legacy source.
- Loaders should accept a `data_root` and `download` flag.

## 8. Development Priorities

**Phase 1 (Foundation)**
- Create `core/` (config, runtime, data, logging).
- Build adapters for:
  - `legacy/006-lora` (cleanest training loop)
  - `legacy/004-diffusion`
  - `legacy/003-clip`

**Phase 2 (Composition)**
- Implement `clip_guided_diffusion` composition.
- Add `transformer_moe` composition.

**Phase 3 (Long Tail)**
- Add RL, GAN, RAG, and single‑file prototypes.
- Normalize tests and add smoke checks.

## 9. Colab Expectations
- All demos should run in <10 minutes by default.
- Include `--fast_demo` mode: tiny dataset, 1–2 epochs.
- Provide clear outputs: plots, samples, and logs in `runs/`.

## 10. Practical Notes from Current Code
- Many legacy READMEs are templates; do not trust them for real instructions.
- Some projects have local `venv/` directories; treat them as artifacts.
- Tests are mostly empty; rely on smoke tests.
- `legacy/010-transformer` already references Colab in README and uses a tokenizer pipeline.

