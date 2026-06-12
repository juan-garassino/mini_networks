# Chapter 05 — Diffusion Models: Denoising as Generation

## Theory recap

A DDPM destroys data with noise, then learns to reverse the destruction. The
**forward process** adds Gaussian noise over T timesteps following a beta schedule;
thanks to the closed form `x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε`, any `x_t` can be
produced in one shot. The model — a UNet — is trained on **epsilon prediction**:
given `x_t` and `t`, predict the noise `ε` that was added, with a plain MSE loss.
Sampling runs the chain backwards: start from pure noise `x_T` and repeatedly apply
the learned reverse step `p(x_{t-1} | x_t)` until `x_0` emerges.

Unlike a GAN there is no adversary, so training is stable — the price is that
sampling needs T sequential network calls.

## In this repo

Code lives in `src/mini_networks/models/diffusion/`. Registry name: `diffusion`.

- **NoiseScheduler** (`models/diffusion/scheduler.py`): precomputes `betas` (linear or
  cosine schedule, set by `DiffusionConfig.schedule`), `alphas_cumprod`, and the
  posterior variance. `add_noise(x0, noise, t)` is the closed-form forward process;
  `step(model_output, t, x_t)` is one DDPM reverse step (deterministic mean at `t == 0`,
  mean + posterior noise otherwise).
- **UNet** (`models/diffusion/model.py`): a small encoder–bottleneck–decoder with
  `ResBlock`s, sinusoidal time embeddings injected per block, and a `SelfAttention`
  layer in the bottleneck. It predicts ε, same shape as the input.
- **Trainer** (`models/diffusion/trainer.py`, `DDPMTrainer`): scales images to `[-1, 1]`,
  samples random `t`, applies `add_noise`, and minimizes `F.mse_loss(pred_noise, noise)`.

### EMA, curriculum, warmup

All three are options on `DiffusionConfig` (`models/diffusion/config.py`):

- **EMA** (`ema_decay`, default 0.9999; 0.0 disables): an `EMA` class keeps a shadow
  copy of the weights updated as `shadow = decay·shadow + (1-decay)·model` each step.
  The shadow is saved as `model_ema.pt` next to `model.pt`, and
  `load_checkpoint()` **prefers `model_ema.pt` when present** — EMA weights give
  sharper, less noisy samples. `infer()` also uses the EMA model when available.
- **Curriculum** (`curriculum: bool`): sorts each batch hardest-first by pixel
  variance (`_image_complexity` / `_sort_batch_by_complexity`).
- **Warmup** (`warmup_steps`): linear LR ramp via `LambdaLR`.

### The shared sample loop

All diffusion-style sampling in the repo goes through one driver:
`sample_loop()` in `src/mini_networks/core/diffusion/sampling.py`. It walks
`t = T-1 … 0`, calling a `predict_noise(x, t_batch, t, state)` function, an optional
`guidance_fn(x, t_batch, t, eps, state)` that can rewrite the predicted noise (this is
how CLIP guidance and classifier guidance plug in), and an optional `step_callback`
that can replace `x` after each step. `DDPMTrainer.infer()` uses it with a plain
lambda; the guided compositions pass real guidance functions.

### Tier-capped timesteps

`BaseConfig.effective_timesteps` (`core/config.py`) caps the chain length by training
tier using the budget table in `core/tiers.py`: **S = 25, M = 200, L = uncapped**
(full `timesteps`, default 1000). The trainer builds its `NoiseScheduler`, draws
training `t`, and runs `sample_loop` all from `config.effective_timesteps` — train and
sample **must share the chain length**, because the schedule's `ᾱ_t` values depend on
T: a model trained on a 25-step chain has never seen the noise levels of a 1000-step
chain, and sampling with mismatched T walks through noise statistics the model
cannot denoise.

## Variants in the repo

- **ConditionedUNet** (`models/diffusion/model.py`): class + time conditioned UNet for
  **classifier-free guidance** — class labels are one-hot embedded and randomly dropped
  with `drop_prob` during training so the model learns both conditional and
  unconditional prediction; at sampling, the two predictions are blended. Used by the
  guided compositions (e.g. `compositions/transformer_clip_diffusion.py`).
- **Latent diffusion** (`compositions/latent_diffusion.py`): runs the same DDPM machinery
  inside a VAE's latent space — see the compositions chapter.
- **tabular_diffusion** (`models/tabular_diffusion/`): the same epsilon-prediction recipe
  on tabular rows. `TabularDenoiser` is a 3-layer MLP (it currently ignores `t`), with a
  minimal `TabularNoiseScheduler` and a simplified ancestral reverse step — a toy that
  shows diffusion is not image-specific.
- **pixelcnn** (`models/pixelcnn/`): the autoregressive contrast. `MaskedConv2d` (mask
  types A/B) ensures each pixel only sees pixels above/left of it — generation order is
  baked into the architecture rather than into a noise chain. The repo's trainer keeps
  it deliberately simple (MSE reconstruction, one-pass refinement in `infer()`).

`diffusion` and `pixelcnn` are `JUDGED_MODELS` in `colab/gate.py`: the quality gate
scores their samples with the classifier judge (confidence × class coverage, see
chapter 04). Thresholds in `core/evalspec.py`: diffusion judge_score ≥ 0.25 (M) /
0.50 (L); pixelcnn ≥ 0.10 (M) / 0.30 (L); tabular_diffusion gates on `eval_loss`.

## Try it

```bash
uv run python main.py train --model diffusion --fast_demo
uv run python main.py sweep --check --models diffusion,tabular_diffusion,pixelcnn --fast_demo
```

## Latest results

<!-- results:start items=diffusion,tabular_diffusion,pixelcnn -->

_Latest sweep: tier S on cpu_

| Item | Status | Metric | Value | Threshold |
|---|---|---|---|---|
| diffusion | pass | judge_score | 0.0223 | n/a |
| tabular_diffusion | pass | eval_loss | 1.1316 | n/a |
| pixelcnn | pass | judge_score | 0.0111 | n/a |

<!-- results:end -->
