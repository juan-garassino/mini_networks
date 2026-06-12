# Chapter 06 — Sequence Models: Recurrence, Attention, State Spaces

## Theory recap

Three philosophies for modeling token sequences:

1. **Recurrence** (RNN/LSTM/GRU): process one token at a time, compressing all history
   into a fixed-size hidden state. O(1) memory per step at inference, but training is
   sequential and long-range information must survive the bottleneck.
2. **Attention** (Transformer): process all positions in parallel; every token can look
   directly at every earlier token. Training parallelizes beautifully, but attention is
   O(T²) and generation re-processes the context window each step.
3. **State spaces** (Mamba): like an RNN, keep a running state with a learned decay —
   but make the recurrence linear so it can (in principle) be computed as a parallel
   scan. Linear in T, no attention, no quadratic blow-up.

All three model the same objective here: next-token prediction with cross-entropy on
character (or BPE) text.

## In this repo

All three families expose the **same interface** so trainers are drop-in compatible:
`logits, aux = model(tokens)` and `model.generate(prompt, max_new_tokens, temperature)`.

### RNN / LSTM / GRU — `src/mini_networks/models/rnn/` (registry: `rnn`)

- `RNNLanguageModel` (`models/rnn/model.py`): `token_embed → nn.RNN|LSTM|GRU → lm_head`.
  The cell is selected by `RNNConfig.cell_type` (`"rnn" | "lstm" | "gru"`, default
  `"lstm"`) via a `_CELL_MAP` lookup — three architectures, one class.
- `forward()` returns `(logits [B, T, V], aux_loss=0.0)` to match the others.
- `generate()` **carries the hidden state between steps**: it warms the state up on the
  prompt once, then feeds back one token at a time, advancing the state — each new token
  costs a single cell step instead of reprocessing the whole context. This is the RNN's
  structural advantage at inference time.

### TransformerLM — `src/mini_networks/models/transformer/` (registry: `transformer`)

- `TransformerLM` (`models/transformer/model.py`): decoder-only, learned token + position
  embeddings, N `TransformerBlock`s (causal `nn.MultiheadAttention` + FFN), LayerNorm,
  LM head. **`forward()` returns `(logits, aux_loss)`** — the trainer adds `aux` to the
  cross-entropy loss (`models/transformer/trainer.py`).
- **Tokenizers** (`models/transformer/tokenizer.py`): character-level `CharTokenizer` or
  byte-level `BPETokenizer` (ids 0–255 are raw bytes; merged tokens start at 256),
  selected by `TransformerConfig.tokenizer_type` (`"char" | "bpe"`). The trainer saves
  `tokenizer.json` with the checkpoint and `load_checkpoint()` auto-detects which kind
  it is, inferring `vocab_size` from the embedding matrix in the state dict.
- **Pluggable FFN** via `TransformerConfig.block_type` (`"standard" | "moe" | "mamba"`),
  built by `_make_ffn()` — every FFN returns `(output, aux_loss)`:
  - **StandardFFN**: Linear → GELU → Linear → Dropout; aux = 0.
  - **MoEFFN**: a Gumbel-softmax `_TopKRouter` picks the top-k of N experts per token;
    an always-on shared path (scaled by a learnable `shared_scale`) is added to the
    routed expert sum. The aux loss is a **balance loss** — KL(mean routing probs ‖
    uniform) minus an entropy bonus — which punishes the router for dumping every token
    on one expert. Knobs: `moe_num_experts`, `moe_top_k`, `moe_balance_loss_weight`,
    `moe_entropy_bonus`, `moe_router_temp`, `moe_add_gumbel`, `moe_shared_scale`.
  - **MambaFFN**: depthwise `Conv1d` for local mixing plus a **gated exponential decay
    scan** `s_t = a·s_{t-1} + b·u_t` with learned per-channel `a` (squashed to (0,1) via
    `exp(-softplus(a))`) and `b`, sigmoid-gated output, internal residual. Knobs:
    `mamba_d_state`, `mamba_d_conv`.

### NanoMamba — `src/mini_networks/models/mamba/` (registry: `mamba`)

- `NanoMamba` (`models/mamba/model.py`): a language model built **entirely** from
  `MambaBlock`s — **no attention anywhere**. Each block: LayerNorm → project to 2×d_model
  → causal depthwise conv → split into (signal, gate) → SiLU/sigmoid → the same gated
  exponential-decay SSM scan → gate, project, dropout, residual. The block plays the
  role attention plays in a Transformer: mixing information across positions.
- Same `(logits, aux_loss=0.0)` contract; `generate()` re-processes the trailing
  `seq_len` window each step, like `TransformerLM.generate()` (the educational fast-path
  of carrying state lives in the RNN chapter above).

## Comparing the three

| | RNN/LSTM/GRU | TransformerLM | NanoMamba |
|---|---|---|---|
| Position mixing | recurrent hidden state | causal self-attention | depthwise conv + decay scan |
| Train-time parallelism | none (sequential) | full (all positions) | conv parallel, scan sequential here |
| Cost per generated token | O(1) (state carry) | O(T) re-process window | O(T) re-process window |
| Long-range memory | squeezed through state | direct lookup | exponential decay |

The quality gate (`core/evalspec.py`) holds all three to a cross-entropy `eval_loss`
threshold on char-level text: transformer ≤ 2.6 (M) / 2.0 (L); rnn and mamba ≤ 2.8 (M)
/ 2.2 (L) — attention earns its keep with a tighter bar.

## Try it

```bash
uv run python main.py train --model transformer --fast_demo
uv run python main.py train --model rnn --fast_demo
uv run python main.py train --model mamba --fast_demo
```

## Latest results

<!-- results:start items=rnn,transformer,mamba -->

_Latest sweep: tier S on cpu_

| Item | Status | Metric | Value | Threshold |
|---|---|---|---|---|
| rnn | pass | eval_loss | 3.8465 | n/a |
| transformer | pass | eval_loss | 3.2333 | n/a |
| mamba | pass | eval_loss | 3.9955 | n/a |

<!-- results:end -->
