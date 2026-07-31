# Chapter 13 — Foundation Minis: Prompts, Fields, Search, Graphs

## Theory recap

Four ideas that define the modern era, each reduced to its irreducible core:

1. **Graphs as a modality** (GCN — this chapter's first entry): learning from
   *structure* when features alone are not enough.
2. **Promptable perception** (mini-SAM, planned): one model, many answers —
   the prompt selects *which* object to segment.
3. **Neural fields** (mini-NeRF, planned): a scene as a continuous function,
   supervised only through rendering.
4. **Search-guided learning** (mini-AlphaZero, planned): a network that
   learns from a planner's improved decisions, then surpasses raw play.

## In this repo

### GCN — `src/mini_networks/models/gnn/` (registry: `gnn`)

- Task: transductive node classification on a seeded **stochastic block
  model** (200 nodes, 4 communities, `p_in=0.15` ≫ `p_out=0.02`), with only
  **5 labeled nodes per community**. Node features carry a deliberately weak
  signal — weak enough that a feature-only classifier stays far below the
  gate bar, so the metric provably measures **message passing**, not feature
  leakage. `evaluate()` logs that baseline (`mlp_baseline_accuracy`) every
  run as the honesty evidence.
- Model: 2 dense-matmul GCN layers — `H' = ReLU(Â X W)` with
  `Â = D^-1/2 (A+I) D^-1/2` (self-loops keep each node's own feature and
  guard isolated nodes). No PyG: the message-passing equation is the code.
- Tier note: the dataset is ONE item (the whole graph), so one epoch is one
  full-batch gradient step — `core/tiers.py` gives gnn 200 M-epochs (each a
  sub-millisecond 200×200 matmul).
- Showcase: community-sorted adjacency heatmap (the block structure is the
  whole story) + per-node predictions.

```bash
uv run python main.py train --model gnn --fast_demo
```

### Mini SAM — `src/mini_networks/models/sam/` (registry: `sam`)

- Task: **two MNIST digits** on a 56x56 canvas (`TwoDigitSamDataset`) — the
  image alone is ambiguous, so the mask depends on the prompt. A click (or
  box) selects the target digit; a negative click can exclude the other one.
- Model: conv encoder → 14x14 tokens, Fourier-feature prompt encoder, 2
  rounds of **two-way attention**, then **3 candidate masks + an IoU head**;
  training backprops only the best head per sample (min-loss), inference
  returns the self-rated best — SAM's ambiguity machinery intact at nano
  scale.
- Honesty evidence: `evaluate()` reports `eval_iou` (clicked digit — the gate
  metric) AND `wrong_prompt_iou` (the same mask scored against the other
  digit). If the model ignored prompts, the two would match; the gap proves
  promptability.
- Showcase: `prompt_variations.png` — same composite, click digit A vs digit
  B, two different masks.

```bash
uv run python main.py train --model sam --fast_demo
```

## Latest results

<!-- results:start items=gnn,sam -->
<!-- results:end -->
