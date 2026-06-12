# 03 — Autoencoders: learning to compress

An autoencoder learns the identity function through a constraint: encode the input
into something smaller (or structured), then decode it back. The constraint forces the
network to keep only what matters — and that compressed representation, the *latent
space*, is the real product. Reconstruction is just the training signal. Latents are
what you classify on, interpolate through, sample from, and (the payoff at the end of
this chapter) run diffusion in.

Both models train on MNIST `classification`-mode images, ignoring the labels.

## unet_ae — UNet as autoencoder

`UNetAutoencoder` is a one-line subclass of the segmentation UNet: `SegUNet` with
`in_channels=1, out_channels=1` (`src/mini_networks/models/unet_ae/model.py`). The
encoder downsamples twice (28 → 14 → 7, channels 32 → 64 → 128 at the bottleneck);
the decoder upsamples with transposed convs; output goes through a sigmoid. The
trainer (`models/unet_ae/trainer.py`) minimizes plain `F.mse_loss(recon, images)`.

The defining feature is the **skip connections**: each decoder level concatenates the
matching encoder feature map before convolving. Fine spatial detail travels straight
across instead of squeezing through the bottleneck, so reconstructions are sharp —
but note the trade-off this teaches: information can bypass the bottleneck, so the
7×7 bottleneck alone is *not* a complete summary of the image. Skips make great
image-to-image models (segmentation, denoising, the diffusion UNet) and weak
compressors. Quality bar (`core/evalspec.py`): `eval_loss` ≤ 0.08 at M, ≤ 0.03 at L.

## vae — variational autoencoder

`ConvVAE` (`src/mini_networks/models/vae/model.py`) makes the latent space
*probabilistic*. The encoder (two stride-2 convs, 1 → 32 → 64, then flatten 64·7·7)
outputs not a point but a distribution per image: `fc_mu` and `fc_logvar` heads,
default `latent_dim=32`. The decoder maps a latent vector back through a linear layer
and two transposed convs to a sigmoid image.

Two ideas carry the model:

1. **Reparameterisation trick** — you can't backprop through "sample from
   N(mu, sigma)". Rewrite the sample as `z = mu + eps * std` with
   `eps ~ N(0, I)` (`ConvVAE.reparameterize`): randomness moves into an input,
   gradients flow to `mu` and `logvar`.
2. **ELBO = reconstruction + KL** — `vae_loss` returns
   `mse_loss(recon, x) + beta * KL(q(z|x) || N(0, I))`, with the KL in closed form:
   `-0.5 * mean(1 + logvar - mu² - exp(logvar))`. Reconstruction keeps latents
   informative; KL packs them into a standard normal so that *any* `z ~ N(0, I)`
   decodes to a plausible digit. `beta` (config field) trades the two off, à la
   beta-VAE.

The trainer logs `loss`, `recon`, and `kl` separately per epoch — watch the KL term
to see the prior being enforced. `infer` accepts `{"sample": n}` to decode random
latents (generation, not reconstruction). Simplified vs the literature: per-pixel
Gaussian likelihood collapsed to mean MSE, no KL warm-up schedule.
Quality bar: `eval_loss` ≤ 220 at M, ≤ 160 at L (recon+KL scale, see the
justification comment in `core/evalspec.py`).

## Why latent spaces: the payoff

Once a VAE gives you a compact latent space, expensive generative models can run
there instead of in pixel space. The `latent_diffusion` composition
(`src/mini_networks/compositions/latent_diffusion.py`) does exactly what Stable
Diffusion does at toy scale: it trains a convolutional VAE
(`models/diffusion/vae.py` — a spatial-latent variant compressing 1×28×28 → 4×7×7,
1/16 the pixels), freezes it, then trains a diffusion UNet to denoise *latents*
rather than images. Sampling runs the reverse diffusion loop in latent space and
decodes the final `z` with the VAE. Note it uses that spatial-map VAE, not this
chapter's vector-latent `ConvVAE` — diffusion UNets want a `[C, H, W]` latent to
convolve over.

Run them:

```bash
uv run python main.py train --model unet_ae --fast_demo
uv run python main.py train --model vae --training_tier M
uv run python main.py compose --composition latent_diffusion --fast_demo
```

## Latest results

<!-- results:start items=unet_ae,vae -->

_Latest sweep: tier S on cpu_

| Item | Status | Metric | Value | Threshold |
|---|---|---|---|---|
| unet_ae | pass | eval_loss | 0.2459 | n/a |
| vae | pass | eval_loss | 0.2519 | n/a |

<!-- results:end -->
