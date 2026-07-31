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

## Latest results

<!-- results:start items=gnn -->
<!-- results:end -->
