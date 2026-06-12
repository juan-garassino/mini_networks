# Chapter 11 — The Composition Layer

## What compositions are

Every model in `src/mini_networks/models/` trains alone. Compositions, under
`src/mini_networks/compositions/`, are **multi-model pipelines**: they train
two or more models (or reuse trained ones) and chain their outputs — a CLIP
encoder steering a diffusion sampler, a segmentation mask becoming a bounding
box, a teacher denoiser distilled into a student. The point is educational:
each model stays simple, and the interesting behaviour emerges from the
wiring. The 19 registered compositions are listed in `COMPOSITIONS` in
`src/mini_networks/colab/launcher.py`, each with a `_run_<name>()` runner
dispatched by `run_composition()`.

## The contract

Every runner returns a **dict containing `run_dir` plus real output tensors**
(or text), e.g. `{"images": ..., "class_id": ..., "config": cfg, "run_dir":
str(logger.run_dir)}`. The quality gate (`src/mini_networks/colab/gate.py`,
behind `main.py sweep --check`) enforces this: `check_composition()` reads
`output["run_dir"]` to find `metrics.jsonl` for the loss-series sanity check,
then `_validate_probe_output()` (in `colab/launcher.py`) strips the metadata
keys (`config`, `run_dir`) and fails the run if what remains is empty — an
empty tensor, empty list, or blank string is a gate failure, not a pass.

## Tour of the families

**Guided generation.** `clip_guided_diffusion.py` trains CLIP and a
`ConditionedUNet` (classifier-free guidance, label-drop `drop_prob=0.1`,
`eps = (1+w)·eps_cond − w·eps_uncond`), optionally a VAE for latent-space
sampling; `text_to_image()` maps a query to the nearest digit class via CLIP
text embeddings, and `dual_oscillation()` is the rotation trick — every
`flip_every` denoising steps, rotate the latent 180° with `torch.rot90(k=2)`
and toggle conditioning between two classes, so the trajectory oscillates
between digits while rotation symmetry preserves spatial structure.
`classifier_guided_diffusion.py` is Dhariwal-style guidance from a `SmallCNN`
classifier; `classifier_guided_gan.py` and `clip_guided_gan.py` add a
classifier or CLIP-similarity term to the generator loss.

**Pipelines.** `transformer_clip_diffusion.py` chains three independently
trained models: the TransformerLM generates `k_prompts` candidate prompts,
CLIP ranks them against per-class text embeddings to pick a digit class, and
the CFG diffusion sampler generates the image. `rag_guided_generation.py`
is retrieve-then-generate (chapter 08); `rag_conditioned_diffusion.py` feeds
the RAG-generated prompt into `CLIPGuidedDiffusion`. `image_captioning.py`
decodes captions from `VisionPatchEncoder` tokens with cross-attention.

**Comparisons.** `gan_diffusion_comparison.py` trains a GAN and a DDPM on
identical data and config, then `compare()` generates from both and reports
a **pixel-variance diversity metric** (`_pixel_variance` — mean per-image
variance) alongside the sample grids, making the GAN-sharp-but-collapsing
vs diffusion-stable-but-slow trade-off measurable.

**Latent diffusion.** `latent_diffusion.py` trains a VAE first, then a UNet
on the VAE's latents (`[B, 4, 7, 7]` instead of `[B, 1, 28, 28]`). Crucially
the VAE is switched to `eval()` and encoded under `torch.no_grad()` before
UNet training — if its BatchNorm statistics or reparameterisation noise kept
shifting, the UNet would chase a moving latent distribution. Sampling runs
the reverse process in latent space and decodes once at the end.

**Multimodal fusion.** `multimodal_fusion_baseline.py` classifies MNIST from
image + caption through `MultiModalEncoder` with pluggable fusion (concat /
gated / cross-attention); `audio_text_dual_encoder.py`,
`tabular_text_dual_encoder.py`, `audio_text_contrastive.py`, and
`tabular_text_cross_attention.py` are the CLIP-style contrastive pairings
from chapter 09, all built on `ContrastiveCompositionBase` in
`compositions/base.py`.

**Vision pipelines.** `segment_then_detect.py` trains a `SegUNet` and derives
bounding boxes from predicted masks with `mask_to_bbox()` — detection without
a detection head. `multitask_vision.py` shares one encoder across
classification, segmentation, and detection heads on a 56×56 canvas dataset.

**Model surgery.** `diffusion_distillation.py` distills a trained UNet
teacher into a `SmallDenoiser` student that mimics its noise predictions.
`lora_lm.py` freezes a pretrained TransformerLM and fine-tunes only a
low-rank `LoRALinear` adapter wrapped around the output head.

(`rl_rlhf_maze.py` — the DQN → LM → PPO bridge — lives in the same package
and is covered in chapter 10.)

## Try it

```bash
uv run python main.py list
uv run python -c "from mini_networks.colab.launcher import run_composition; \
run_composition('gan_diffusion_comparison', fast_demo=True)"
```

## Latest results

<!-- results:start items=clip_guided_diffusion,transformer_clip_diffusion,gan_diffusion_comparison,latent_diffusion -->

_Latest sweep: tier S on cpu_

| Item | Status | Metric | Value | Threshold |
|---|---|---|---|---|
| clip_guided_diffusion | pass | n/a | n/a | n/a |
| transformer_clip_diffusion | pass | n/a | n/a | n/a |
| gan_diffusion_comparison | pass | n/a | n/a | n/a |
| latent_diffusion | pass | n/a | n/a | n/a |

<!-- results:end -->
