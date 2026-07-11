"""Tier budget table — the single visible place training budgets live.

BaseConfig.effective_* properties read from here. `None` means uncapped.
Per-model overrides are added during Phase 2 stabilization triage; each
override carries a comment saying why that model needs a different budget.
"""
from __future__ import annotations

DEFAULTS: dict[str, dict[str, int | None]] = {
    #      epochs  batch_cap  sample_limit  train_batches  eval_batches  timesteps
    "S": {"epochs": 1, "batch_cap": 16, "sample_limit": 32,
          "train_batches": 1, "eval_batches": 1, "timesteps": 25},
    # M re-sized 2026-07-10 for the L4 cloud sweep (was 3/32/512/8/4 — a Colab
    # wall-time compromise under which deep vision nets sat at chance accuracy
    # after 24 steps; L4 tasks finished in ~25s, so there was huge headroom).
    "M": {"epochs": 5, "batch_cap": 64, "sample_limit": 4096,
          "train_batches": 100, "eval_batches": 8, "timesteps": 200},
    "L": {"epochs": None, "batch_cap": None, "sample_limit": None,
          "train_batches": None, "eval_batches": None, "timesteps": None},
}

# model_name -> tier -> partial budget override (merged over DEFAULTS)
MODEL_OVERRIDES: dict[str, dict[str, dict[str, int | None]]] = {
    # judge trajectory without EMA: 0.023 @ 24 steps, 0.048 @ 500, 0.139 @ 2k,
    # 0.049 @ 3.5k (non-monotone). With generator EMA, run long and let the
    # slow average ride out the oscillations (m-full-2). sample_limit
    # uncapped: 50 epochs over the SAME 4096 images let D memorize and G
    # chase artifacts (judge 0.14 with textured-noise samples, m-vision-10);
    # step count is still bounded by train_batches, this only adds variety.
    "gan": {"M": {"epochs": 80, "sample_limit": None}},
    # Guided-diffusion compositions match base diffusion's 10 epochs: at the
    # 5-epoch default their CFG UNets stay noisy and guide_weight=2 AMPLIFIES
    # the error (eps_c + w*(eps_c - eps_u)) — samples were noise while base
    # diffusion produced digits on the same image (m-vision-1/2).
    "clip_guided_diffusion": {"M": {"epochs": 10}},
    "transformer_clip_diffusion": {"M": {"epochs": 10}},
    "rag_conditioned_diffusion": {"M": {"epochs": 10}},
    "classifier_guided_diffusion": {"M": {"epochs": 10}},
    # Self-supervised needs more steps than supervised: kNN 0.16 at 5 epochs
    # with healthy loss — representation quality lags the loss (m-vision-1).
    "dino": {"M": {"epochs": 15}},
    # Char-level LMs are fast (~1s/epoch on L4) and generation quality lags
    # eval_loss badly — the audit graded all of them "gibberish words, learned
    # structure only". Triple the budget for legible text.
    "transformer": {"M": {"epochs": 15}},
    "moe": {"M": {"epochs": 15}},
    "mamba": {"M": {"epochs": 15}},
    "rnn": {"M": {"epochs": 15}},
    "rag": {"M": {"epochs": 15}},
    "text_seq2seq": {"M": {"epochs": 15}},
    "rlhf": {"M": {"epochs": 15}},
    "dpo": {"M": {"epochs": 15}},
    # 30, not 15: at 15 the captioner produced grammatical captions with the
    # WRONG digit ("the number 8" for a 5) — the cross-attention content
    # signal trails the language model (m-full-4 vision audit).
    "image_captioning": {"M": {"epochs": 30}},
    # pixelcnn samples were stroke fragments at 5 epochs (judge 0.6-0.77);
    # AR likelihood models keep improving with budget.
    "pixelcnn": {"M": {"epochs": 15}},
    # judge 0.31 and 0.18 on identical configs @ 5 epochs — the metric sits ON
    # the 0.25 bar; double the steps for margin (m-full-2).
    "diffusion": {"M": {"epochs": 10}},
    # Honest FSDD (random subset, all 10 classes) needs real training: the old
    # 1.0 came from a head-sliced 2-class subset (m-full-2). The raw-waveform
    # CNN is the slowest learner in the zoo: 0.094 @ 5 ep -> 0.383 @ 15
    # (m-triage-5); epochs cost ~1.5s each on L4.
    "audio_classifier": {"M": {"epochs": 40}},
    "audio_spectrogram": {"M": {"epochs": 15}},
    "audio_melspectrogram": {"M": {"epochs": 15}},
    "audio_transformer": {"M": {"epochs": 15}},
    # 0.71 accuracy @ 5 epochs: depthwise-separable blocks train slower than
    # plain convs; double the epochs.
    "mobilenet": {"M": {"epochs": 10}},
    # 0.55 accuracy @ 5 epochs on a tiny tabular set where epochs cost
    # milliseconds — let it actually converge.
    "tabular_classifier": {"M": {"epochs": 40}},
}


def budget(model_name: str, tier: str, key: str) -> int | None:
    override = MODEL_OVERRIDES.get(model_name, {}).get(tier, {})
    if key in override:
        return override[key]
    return DEFAULTS[tier][key]
