# Mini Kimi K3 + Mini DeepSeek V4 — design

Date: 2026-07-30
Status: approved (brainstormed in-session; papers on the owner's Desktop)
Sources: Kimi K3 technical report (arXiv 2607.24653), DeepSeek V4 technical report (arXiv 2606.19348)

## Goal

Add two frontier-LM architectures as models 36 and 37 of the zoo, both on the
TransformerLM char-level lineage (identical tiny-Shakespeare corpus, tokenizer,
trainer contract) so the results table becomes a 4-way architecture comparison:
dense transformer → moe → kimi → deepseek.

The papers are two contrasting answers to the same three problems — that
contrast is the curriculum payload:

| Problem | Kimi K3 (`kimi`) | DeepSeek V4 (`deepseek`) |
|---|---|---|
| Attention cost | KDA linear delta-rule attention, 3:1 hybrid with gated full attention | KV compression: CSA (compress m→1 + top-k sparse) and HCA (heavy compress, dense), + sliding window + attention sink |
| Residual bottleneck over depth | AttnRes: softmax attention over previous block outputs | mHC: residual stream widened to n×d, Sinkhorn doubly-stochastic mixing |
| Sparse FFN | Stable LatentMoE (experts in latent space) + SiTU-GLU + bias balancing | DeepSeekMoE fine-grained + shared experts, sqrt(softplus) affinity, bias balancing |

## Shape

Both follow the `moe` reuse idiom: `Config(TransformerConfig)`,
`Trainer(TransformerTrainer)` overriding only `_build`, registry entry reusing
`make_transformer_dataloader`. Models keep the `forward(tokens) -> (logits,
aux_loss)` contract, a `generate()` method, and name their embedding
`token_embed` (checkpoint vocab inference relies on it).

## kimi — components

- **KDA layer** (paper §2.1.1): recurrent delta rule
  `S_t = (I − β k kᵀ) Diag(α) S_{t−1} + β k vᵀ`, per-channel decay
  `α = exp(g_min · sigmoid(exp(A_h)·z))` with `g_min = −5` (the K3
  lower-bounded decay, Eq. 5). q/k = L2Norm(SiLU(causal depthwise conv)),
  β = sigmoid per head, head-wise RMSNorm + full-rank sigmoid output gate.
  Plain time loop — the chunkwise/Tensor-Core form is a scale artifact.
- **Hybrid 3:1** (§2.1): per block, 3 KDA + 1 gated full attention with NoPE
  (no positional embeddings anywhere — KDA conv+decay carries position).
  Plain MHA stands in for MLA (latent-KV compression is V4's lesson; keeping
  it out keeps the contrast clean). Extra final global layer per the paper.
- **Block AttnRes** (§2.2): learnable pseudo-query per layer over
  [embedding, previous block sums, running partial sum], RMSNorm on keys.
- **Stable LatentMoE** (§2.3): shared full-width expert + routed experts in a
  latent space (W_down → experts → RMSNorm → W_up), SiTU-GLU
  (β₁=4, β₂=25, Eq. 12), aux-loss-free sign-rule bias balancing (quantile
  balancing's histogram estimator is a distributed-training artifact).

## deepseek — components

- **TokenCompressor** (§2.3, Eqs. 20-23): softmax(Z+B)-weighted pooling of
  every m tokens; queries attend only to strictly-preceding compressed blocks.
- **CSA layer**: latent query down/up projection, ReLU-scored lightning
  indexer → top-k compressed entries → shared-KV MQA (entry is both K and V),
  + sliding-window branch (n_win raw tokens) + learnable per-head attention
  sink in the softmax denominator. Grouped output projection skipped (width
  artifact).
- **HCA layer**: same machinery, m′ ≫ m, dense (no indexer).
- **Partial RoPE** on last dims of q and compressed entries; inverse RoPE
  (position −i) on outputs so they carry relative position (§2.3.3).
- **mHC** (§2.2): residual stream n_hc×d; dynamic+static A/B/C mappings,
  A = σ, C = 2σ, B = Sinkhorn(exp(·)) doubly stochastic.
- **DeepSeekMoE FFN**: fine-grained routed + shared experts, sqrt(softplus)
  affinity, sign-rule bias balancing, SwiGLU experts.

## Extra paper tricks included (second read, owner request)

- **MTP** in BOTH models (K3 Table 1 and V4 both ship 1 MTP layer): predict
  token t+2 from h_t + emb(x_{t+1}); the CE rides the aux-loss channel so the
  trainer contract is untouched.
- kimi: **1 dense first layer** (K3 Table 1) and **2 shared experts**.
- deepseek: **hash-routed MoE** in the first block (token-id hash, Roller et
  al.), **sequence-wise balance loss** (slight, V3-style), mHC applied **per
  sublayer** (attention and FFN each wrapped, Eq. 1 branch form).
- Neither model has absolute position embeddings: kimi is fully NoPE;
  deepseek uses partial RoPE + inverse-RoPE'd outputs only.

## Skipped from both papers (scale artifacts, out of zoo scope)

Muon / Per-Head Muon optimizer, FP8/FP4 precision schemes, vision pathways
(MoonViT-V2), million-token context machinery, chunkwise KDA kernels,
grouped output projection, quantile-balancing histogram estimator.

## Gate & rollout

EvalSpec `_loss(2.8, 2.2)` for both — provisional twins of moe/mamba until
the first M cloud sweep sets honest bars (threshold-honesty rule). S-tier gate
runs in CI (`sweep-s` auto-includes them); M bars + champion registration via
`make -C infra/gcp sweep TIER=M ITEMS=kimi,deepseek`.

## Tests

Per-model smoke (train/eval/infer/checkpoint at S) plus mechanism tests:
KDA decay range + causality, SiTU-GLU boundedness, compressor causality
(no future leakage), Sinkhorn doubly-stochastic property.
