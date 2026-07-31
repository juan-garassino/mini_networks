# Chapter 12 — Generalization & Geometry: When and Why Networks Generalize

## Theory recap

The classical picture — "more parameters ⇒ more overfitting, stop training when
validation loss turns up" — fails in instructive ways on modern networks. This
chapter collects zoo items that each break one piece of the classical story:

1. **Grokking** (Power et al., arXiv 2201.02177): generalization can arrive as a
   *phase transition* long after the training set is perfectly memorized.
2. **Residual Pathway Priors** (arXiv 2112.01388, planned): inductive bias works
   better as a *soft preference* than a hard architectural constraint.
3. **Mode connectivity / loss simplexes** (arXiv 2102.13042, planned):
   independently trained solutions live in connected low-loss volumes, and
   sampling those volumes gives cheap ensembles.
4. **Double descent** (arXiv 2503.02113, planned): test error can fall *again*
   past the interpolation threshold — in networks and classical models alike.

## In this repo

### Grokking — `src/mini_networks/models/grokking/` (registry: `grokking`)

- Task: modular division `a / b (mod 97)` shown as 4-token sequences
  `[a, op, b, =]`; every valid pair is enumerated and split 50/50 into
  train/val **at the pair level** (`core/data/registry.py::ModularArithmeticDataset`)
  — validation pairs are never seen in training, and there is no way to
  interpolate them "locally": the model must find the modular structure.
- Model: the zoo's smallest — a 2-layer causal transformer
  (`GrokkingTransformer`), answer classified from the last position.
- Trainer: **step-based**, not epoch-based (`limit_steps`, the RL idiom):
  S ≈ 50 steps (smoke), M ≈ 20k, L = 100k. The two key knobs are
  `weight_decay=1.0` (AdamW — without it the jump can take orders of magnitude
  longer) and near-full-batch updates (`batch_size=512`).
- The deliverable is the **metrics curve itself**: `train_accuracy` saturates
  early; `val_accuracy` sits at chance ~1/97, then jumps. Watch it live in the
  playground, or read `runs/grokking/<ts>/metrics.jsonl`.

```bash
uv run python main.py train --model grokking --fast_demo   # S smoke
# the real curve needs the M/L step budget (cloud sweep)
```

### Residual Pathway Priors — `src/mini_networks/models/rpp_classifier/` (registry: `rpp_classifier`)

- Each block sums a **constrained** pathway (3x3 conv — translation
  equivariant) and a **free** pathway (dense linear over the flattened feature
  map, which strictly contains the conv as a special case). The preference for
  structure lives in the PRIOR, not the architecture: a weak L2 penalty on the
  conv path, a strong one on the free path — a MAP estimate under Gaussian
  priors of different widths (`RPPClassifierTrainer._loss`).
- `pathway_norms()` reports which pathway carried the solution; on clean MNIST
  the conv path should dominate (the showcase prints both norms).
- Probe the soft-vs-hard bias story with the dataset flavors:
  `--dataset kmnist` / `--dataset tri_mnist` — where the symmetry assumptions
  bend, the free path earns its keep.

## Latest results

<!-- results:start items=grokking,rpp_classifier -->
<!-- results:end -->
