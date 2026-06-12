# Chapter 04 — GANs: Adversarial Training

## Theory recap

A Generative Adversarial Network is a two-player game. A **Generator** G maps random
noise `z` to fake images; a **Discriminator** D looks at an image and outputs the
probability it is real. D is trained to say 1 on real data and 0 on G's output;
G is trained to make D say 1 on its fakes. Both objectives use binary cross-entropy.
At the (theoretical) equilibrium, G's samples match the data distribution and D
outputs ~0.5 everywhere — it can no longer tell real from fake.

The catch: nothing in the loss forces G to cover the *whole* data distribution.
G can win by producing one very convincing digit over and over. This failure is
**mode collapse**, and it is why evaluating a GAN by loss alone is misleading —
you must also measure sample *diversity*.

## In this repo

Code lives in `src/mini_networks/models/gan/`. Registry name: `gan`.

- **Generator** (`models/gan/model.py`): an MLP, noise `[B, 100]` → 256 → 512 → 1024
  → 784, each hidden layer followed by `LeakyReLU(0.2)`, final layer `Tanh`. Output is
  reshaped to `[B, 1, 28, 28]`. Tanh means samples live in `[-1, 1]`; `infer()` rescales
  to `[0, 1]` with `(samples + 1) / 2`.
- **Discriminator** (`models/gan/model.py`): the mirror MLP, flattened image (784) →
  1024 → 512 → 256 → 1, with `LeakyReLU(0.2)` + `Dropout(0.3)` after every hidden layer
  and a final `Sigmoid`. Dropout keeps D from memorizing and overpowering G early.
- **Losses** (`models/gan/model.py`): `gan_d_loss` scores real images against target 1
  and fakes against target 0 with `nn.BCELoss`. Crucially it calls `fake.detach()` —
  the D step must not backpropagate into G's weights. `gan_g_loss` scores fakes against
  target 1 (G wants D fooled), this time *without* detach so gradients flow into G.
- **Trainer** (`models/gan/trainer.py`, `GANTrainer`): per batch, two alternating steps
  with separate `Adam(lr, betas=(0.5, 0.999))` optimizers. First the D step on
  `gan_d_loss(D, real, fake)`; then the G step generates a *fresh* `fake2 = G(z2)` to
  avoid reusing a stale graph, and updates on `gan_g_loss(D, fake2)`. Logs `d_loss` and
  `g_loss` per epoch; saves `generator.pt` and `discriminator.pt`.
- **Evaluation** (`GANTrainer.evaluate`): reports `mean_real_score` — D's mean output on
  real images, which should drift toward ~0.5 as the game balances.
- **Checkpoints**: `load_checkpoint()` rebuilds both nets and loads
  `generator.pt` + `discriminator.pt` from the run's `artifacts/` directory.

## Mode collapse and the judge

GAN losses oscillate by design (the two players push against each other), so the
quality gate does not threshold the loss. Instead, `gan` is one of the
`JUDGED_MODELS` in `src/mini_networks/colab/gate.py`, scored by a trained classifier:

- `judge_samples()` in `colab/gate.py` draws samples from the trained generator
  (16 at S tier, 64 at M/L), feeds them through an MNIST classifier (trained or
  reloaded lazily by `JudgeContext`), and computes:

  ```
  judge_score = mean max-softmax confidence × digit-class coverage
  ```

- **Confidence** asks "do these look like *some* digit?" — blurry mush scores low.
- **Coverage** is `len(set(predicted classes)) / 10` — it directly punishes mode
  collapse. A generator that only emits perfect 7s gets confidence ≈ 1.0 but
  coverage 0.1, so judge_score ≈ 0.1 and the gate fails.

Thresholds live in `src/mini_networks/core/evalspec.py`: `gan` must reach
judge_score ≥ 0.15 at M tier and ≥ 0.40 at L tier, with S-tier sanity checks on
the `g_loss`/`d_loss` series (finite values; trend check where the series allows it).

## Try it

```bash
uv run python main.py train --model gan --fast_demo
uv run python main.py sweep --check --models gan --fast_demo
```

See also `compositions/gan_diffusion_comparison.py`, which trains this GAN and a
DDPM on the same data and compares sample diversity (chapter on compositions).

## Latest results

<!-- results:start items=gan -->

_Latest sweep: tier S on cpu_

| Item | Status | Metric | Value | Threshold |
|---|---|---|---|---|
| gan | pass | judge_score | 0.0108 | n/a |

<!-- results:end -->
