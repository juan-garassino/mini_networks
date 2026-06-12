# Design: Ultimate Mini-Networks Educational Resource with Verified Training

Date: 2026-06-12
Status: Approved by Juan (sections 1–4 approved in brainstorming session)

## Goal

Make this repo a personal reference lab where **every** model and composition demonstrably
learns (per-model eval thresholds + correct inference), trained via Python CLI or
notebooks running on a Colab GPU kernel from VSCode, with the educational substance
living in a docs/ curriculum and annotated source code.

## Decisions (from brainstorming)

- **Audience:** Juan, as a reference lab. Training driven by scripts and notebooks
  (VSCode connected to a Colab kernel for GPU).
- **Learning bar:** per-model quality thresholds verified by evaluation, plus inference
  that is correct (validated output).
- **Coverage:** everything must pass — all ~20 models and all compositions.
- **Foundation repair is Phase 0** of this design (not a separate effort).
- **Educational layer:** docs/ as ordered curriculum + annotated source as textbook.
  Notebooks are training drivers, not the primary teaching artifact.
- **Architecture:** extend the existing tier/sweep system into a quality gate
  (Approach A) — no separate evals/ framework.
- **Legacy:** full git history rewrite to purge legacy/ blobs (force-push accepted).

## Phase structure

Each phase leaves the repo in a working state.

### Phase 0 — Foundation

1. Fix `tests/test_core.py:9-13` stale imports (`MNISTBinarySegmentation` →
   `BinarySegmentationFromDigits`, `MNISTDetection` → `DigitDetection`,
   `MNISTMulticlassSegmentation` → `MulticlassSegmentationFromDigits`) so the suite
   collects and passes again.
2. Fix audit-confirmed training-correctness bugs:
   - `compositions/latent_diffusion.py:75` — VAE never set to `eval()` before encoding
     for UNet training (reparameterise injects noise).
   - `compositions/clip_guided_diffusion.py:344,413` — VAE (BatchNorm) not in eval mode
     during `sample()` / `dual_oscillation()`.
   - `models/rlhf/trainer.py:172` — verify KL direction; fix if inverted.
   - `core/runtime.py:41` and all other `torch.load` call sites — `weights_only=True`.
3. Replace `print()` with the Logger / stdlib logging in the 8 offending trainers
   (gan, lora, transformer, rnn, mamba, rag, rlhf) and `core/data/registry.py`.
4. Hygiene: `git filter-repo` history rewrite to purge `legacy/` venvs and binary data;
   keep legacy source files only if still referenced; extend `.gitignore`; force-push.
5. Makefile gains `test` / `test-ci`; minimal GitHub Actions CI: pytest +
   smoke-import job + S-tier micro-sweep on CPU.

### Phase 1 — Quality gate

**EvalSpec**, registered per model/composition alongside its config in the registry:

```python
@dataclass
class EvalSpec:
    metric: str                      # e.g. "val_accuracy", "perplexity", "iou"
    thresholds: dict[str, float]     # {"M": 0.95, "L": 0.97} — S has no metric bar
    higher_is_better: bool = True
    infer_check: Callable            # validates infer() output: shape/dtype/content
```

**Tier semantics:**

| Tier | Hardware | Bar |
|---|---|---|
| S | CPU (CI) | training completes, no NaN/divergence, final loss < initial loss, `infer_check` passes |
| M | Colab GPU | metric clears threshold — the "actually learns" bar |
| L | Colab GPU | stretch threshold + longer budget; optional per model |

**`main.py sweep --check`** extends the existing sweep. Per item: train at tier →
`evaluate()` → compare metric vs threshold → run `infer_check` on the saved checkpoint
via `load_checkpoint` (verifying the save/load round-trip) → write
`runs/sweep/<timestamp>/report.{md,json}` (pass/fail, metric, threshold, duration).
Non-zero exit if anything fails (CI-able). A `--only <model>` flag re-runs one item.

**Per-model metrics:**

| Model(s) | Metric |
|---|---|
| classifier, resnet, vit, mobilenet, convnext | val accuracy |
| segmentation (binary/multiclass) | IoU |
| detection | IoU + label accuracy |
| transformer, mamba, rnn, rag | val loss / perplexity |
| vae, unet_ae | reconstruction error |
| diffusion, gan, pixelcnn | samples scored by the trained MNIST classifier (digit-confidence + diversity) |
| clip | text↔image retrieval accuracy |
| lora | fine-tune accuracy lift on FashionMNIST |
| rl_maze | success rate |
| rlhf | reward improvement over pretrained LM |
| compositions | their `compare()` outputs / composition-specific checks |

**Error handling:** a model that crashes or times out is reported as `error` with a
traceback excerpt — never silently skipped; the sweep continues.

### Phase 2 — Stabilization to green

Run from `colab/notebooks/00_sweep.ipynb` (clone → `uv sync` → sweep → render report
inline) on the Colab GPU kernel.

1. Baseline: `sweep --check --training_tier M` over everything → first honest report.
2. Triage loop, worst-first, one model at a time: diagnose (LR, schedule, init,
   tier budget, architecture bug) → fix → `sweep --check --only <model>` → commit
   when green. Each fix gets a one-line "what it took to make this train" note that
   lands in the model's docs chapter.
3. **Threshold honesty rule:** thresholds may change during triage only with a written
   justification recorded alongside the threshold — no silent bar-lowering.
4. Tier budgets (epochs/steps per tier per model) consolidate into one `tiers.py`
   table — every model's budget visible and editable in one place.
5. Exit criterion: full M-tier sweep green; S-tier sweep green in CI on every push.

### Phase 3 — Educational layer

**docs/ as curriculum** — restructure into ordered chapters:

```
docs/
  00-overview.md          # repo map, runtime contract, tiers, how to run anything
  01-data.md              # registry, MNIST modes, synthetic datasets
  02-classifiers.md       # CNN, ResNet, ViT, MobileNet, ConvNext
  03-autoencoders.md      # unet_ae, VAE
  04-gan.md
  05-diffusion.md         # DDPM, EMA, curriculum, latent & guided variants
  06-sequence-models.md   # RNN/LSTM/GRU, Transformer, MoE, Mamba
  07-lora-finetuning.md
  08-rag.md
  09-clip-multimodal.md
  10-rl-rlhf.md
  11-compositions.md
```

Each chapter: short theory recap, link to annotated source, results table
auto-generated from the latest sweep report, and Phase 2 stabilization notes.
`scripts/render_results.py` injects sweep results between markers in the chapters so
docs never drift from real runs.

**Annotated source as textbook:** every `model.py` gets a header docstring
(architecture sketch, key equations, deliberate simplifications vs the paper) plus
inline notes only where code is non-obvious (Mamba gated decay scan, CFG in the
conditioned UNet). Trainers stay lean; pedagogy lives with model definitions.

## Testing

- Unit tests for EvalSpec resolution, report generation, and `render_results.py`
  (fast, in CI).
- The S-tier `sweep --check` run is the integration test, in CI on every push.
- Existing test suite must stay green throughout; Phase 0 unblocks it.

## Out of scope

- API hardening beyond `weights_only=True` (auth, rate limits, job persistence) —
  educational/local tool; audit findings recorded but not part of this design.
- New models or compositions.
- Notebook-as-eval; notebooks remain drivers/deep-dives only.

## Docs to update (executed within phases, same commits as the code)

- `CLAUDE.md` — full rewrite: actual `src/mini_networks/` architecture, tier system,
  EvalSpec, sweep workflow; delete the legacy unification plan (Phase 1).
- `AGENTS.md` — same correction if it mirrors CLAUDE.md (Phase 1).
- `README.md` — add `sweep --check`, curriculum index, VSCode→Colab kernel workflow
  (Phase 1, refreshed in Phase 3).
- `docs/README.md` — becomes the chapter index (Phase 3).
- `Makefile` help text — new `test` / `test-ci` / sweep targets (Phase 0).
