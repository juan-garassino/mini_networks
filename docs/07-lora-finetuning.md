# Chapter 07 — LoRA: Low-Rank Adaptation

## Theory recap

Fine-tuning a pretrained model normally updates every weight. LoRA's bet: the *change*
a fine-tune needs, ΔW, is approximately low-rank — it can be written as the product of
two thin matrices, `ΔW = B·A`, with `A: [rank, in]` and `B: [out, rank]`. Freeze the
original `W`, train only A and B, and add their product to the layer's output:

```
forward(x) = W·x + (x · Aᵀ · Bᵀ) · (alpha / rank)
```

Parameter count drops from `O(in·out)` to `O(rank·(in+out))` — for a 512→128 layer at
rank 4, that's ~65k frozen params vs ~2.5k trainable. Initializing `B` to zero means
the adapter starts as an exact identity: at step 0 the model behaves precisely like the
pretrained one, and the fine-tune deviates only as gradients demand.

**What rank trades off**: rank is the expressiveness budget of ΔW. Low rank → fewer
trainable params, less overfitting on small target datasets, but the adaptation can only
move the weights along a few directions. High rank → more expressive ΔW, approaching a
full fine-tune in capacity (and cost). `alpha/rank` scaling keeps the adapter's output
magnitude roughly stable as you change rank, so `alpha` and learning rate don't have to
be retuned with every rank experiment.

## In this repo

Code lives in `src/mini_networks/models/lora/`. Registry name: `lora`.

- **LoRALinear** (`models/lora/model.py`): a Linear layer with a built-in adapter.
  Base `weight` is Kaiming-initialized; `lora_A [rank, in]` is Kaiming-initialized,
  `lora_B [out, rank]` is **zero-initialized**. Forward is exactly
  `F.linear(x, W, bias) + (x @ lora_A.T @ lora_B.T) * (alpha / rank)`.
  `freeze_base()` / `unfreeze_base()` toggle `requires_grad` on `W` and the bias.
- **LoRACNN** (`models/lora/model.py`): the host network —
  Conv1 (1→16) → pool → Conv2 (16→32) → pool → `LoRALinear` FC1 (1568→hidden) →
  `LoRALinear` FC2 (hidden→10). Only the two FC layers carry adapters.
  - `freeze_for_finetune(freeze_conv=True)`: freezes both conv layers (optionally) and
    the base weights of FC1/FC2, leaving **only the LoRA A, B matrices trainable**.
  - `trainable_params()` returns just the parameters with `requires_grad=True`, which
    is what the fine-tune optimizer receives.
- **Two-stage trainer** (`models/lora/trainer.py`, `LoRATrainer.train`):
  1. **Stage 1 — pretrain on MNIST**: `model.unfreeze_all()`, Adam over *all*
     parameters, epochs from `config.pretrain_epochs` (tier-capped via
     `config.tier_epochs`). Logs `pretrain_loss` with `stage: 1`.
  2. **Stage 2 — fine-tune on FashionMNIST**: `model.freeze_for_finetune(...)`, a fresh
     Adam over `model.trainable_params()` **only** — i.e. only A and B (and the convs
     stay frozen when `config.freeze_conv` is true). Logs `finetune_loss` with
     `stage: 2`, epoch indices offset after stage 1.

  The dataset switch happens inside the trainer via
  `make_lora_dataloader(config, dataset="mnist" | "fashion_mnist")` — same loader
  factory, different dataset name. One checkpoint, `model.pt`, holds base weights and
  adapters together.
- **Config** (`models/lora/config.py`): `lora_rank`, `lora_alpha`, `pretrain_epochs`,
  `finetune_epochs`, `freeze_conv`, `hidden_dim`.
- **Evaluation**: standard classification — `evaluate()` returns `eval_loss` and
  `accuracy`. The quality gate (`core/evalspec.py`) requires accuracy ≥ 0.60 (M) /
  0.80 (L), with the S-tier trend check watching `pretrain_loss`/`finetune_loss`.

## Why this is the interesting experiment

The two stages make the LoRA claim falsifiable on a laptop: MNIST digits and
FashionMNIST clothes are different domains, yet stage 2 reaches useful accuracy while
touching only a few thousand adapter parameters — the frozen conv features and frozen
FC weights carry over, and the low-rank ΔW steers them. Try `lora_rank=1` vs
`lora_rank=16` and watch what the rank budget buys. For LoRA on a language model
instead of a CNN, see `compositions/lora_lm.py`.

## Try it

```bash
uv run python main.py train --model lora --fast_demo
uv run python main.py sweep --check --models lora --fast_demo
```

## Latest results

<!-- results:start items=lora -->

_Latest sweep: tier S on cpu_

| Item | Status | Metric | Value | Threshold |
|---|---|---|---|---|
| lora | pass | accuracy | 0.1875 | n/a |

<!-- results:end -->
