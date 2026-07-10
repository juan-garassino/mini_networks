"""Per-model evaluation specs: the quality gate's source of truth.

One entry per model and composition. Thresholds apply at M/L tier only —
S tier checks that training completes, losses are finite, and the loss
trend points down (see colab/gate.py). Threshold changes require a
justification comment next to the number (threshold honesty rule,
docs/superpowers/specs/2026-06-12-ultimate-educational-resource-design.md).

Initial thresholds are deliberately conservative starting points; Phase 2
stabilization tunes them against real M-tier runs on GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalSpec:
    metric: str | None                      # key in evaluate() dict; None = S-style checks only
    thresholds: dict[str, float] = field(default_factory=dict)  # {"M": x, "L": y}
    higher_is_better: bool = True
    score_fn: str | None = None             # named scorer in gate.SCORE_FNS, overrides evaluate()
    loss_keys: tuple[str, ...] = ("loss",)  # metrics.jsonl keys for the S-tier trend check
    s_mode: str = "decreasing"              # "decreasing" | "finite"
    roundtrip: bool = True                  # load_checkpoint round-trip (compositions: False)


def _acc(m: float, l: float, **kw) -> EvalSpec:
    return EvalSpec(metric="accuracy", thresholds={"M": m, "L": l}, **kw)


def _loss(m: float, l: float, **kw) -> EvalSpec:
    kw.setdefault("higher_is_better", False)
    return EvalSpec(metric="eval_loss", thresholds={"M": m, "L": l}, **kw)


def _judge(m: float, l: float, **kw) -> EvalSpec:
    return EvalSpec(metric="judge_score", thresholds={"M": m, "L": l},
                    score_fn="judge_samples", **kw)


def _composition(metric: str | None = None, **kw) -> EvalSpec:
    kw.setdefault("roundtrip", False)
    return EvalSpec(metric=metric, **kw)


EVAL_SPECS: dict[str, EvalSpec] = {
    # ------------------------------------------------------------- models
    "classifier":            _acc(0.85, 0.95),
    "resnet":                _acc(0.85, 0.95),
    "vit":                   _acc(0.75, 0.90),   # ViTs need more data; lower bar at M
    "mobilenet":             _acc(0.80, 0.93),
    "convnext":              _acc(0.80, 0.93),
    "tabular_classifier":    _acc(0.75, 0.90),   # provisional post-split-fix: 30-row honest eval, 1 miss = -3.3%
    # PROVISIONAL (2026-07-11): every pre-split-fix audio number was TRAINING
    # accuracy (FSDD ignored `split` — train==eval; gate audit). Bars below
    # are placed under the expected honest-eval bands and must be re-derived
    # from the first post-fix sweep. audio_classifier's raw-waveform lesson
    # (wrong representation vs spectrograms) still holds.
    "audio_classifier":      _acc(0.20, 0.50),
    "audio_spectrogram":     _acc(0.35, 0.70),
    "audio_transformer":     _acc(0.40, 0.75),   # provisional post-split-fix (0.99 was memorized train acc)
    "audio_melspectrogram":  _acc(0.40, 0.70),   # provisional post-split-fix (was 0.50 on leaked eval)
    "segmentation":          EvalSpec(metric="eval_iou", thresholds={"M": 0.55, "L": 0.75}),
    "detection":             EvalSpec(metric="eval_accuracy", thresholds={"M": 0.55, "L": 0.80}),
    "lora":                  _acc(0.60, 0.80, loss_keys=("loss", "pretrain_loss", "finetune_loss")),
    "clip":                  _loss(2.5, 1.5, loss_keys=("clip_loss", "loss")),
    "simclr":                _loss(4.0, 3.0),
    # s_mode=finite: self-distillation CE is non-monotone early (center warm-up
    # + teacher temperature sharpening push it up before it comes down) — the
    # trend check flagged healthy runs (m-full-3, eval_loss 0.52 with a 4.2
    # bar). Quality is gated by the eval_loss threshold. Bars: uniform
    # baseline is ln(64)≈4.16; observed 0.50-1.03 across M runs.
    "dino":                  _loss(4.2, 3.5, s_mode="finite"),
    "vision_embed":          _loss(4.0, 3.0),
    "transformer":           _loss(2.6, 2.0),    # char-level Shakespeare CE
    "mamba":                 _loss(2.8, 2.2),
    "rnn":                   _loss(2.8, 2.2),
    "rag":                   _loss(2.8, 2.2),
    "rlhf":                  _loss(3.0, 2.4, loss_keys=("pretrain_loss", "ppo_loss")),
    "text_seq2seq":          _loss(2.5, 1.8),
    "text_token_classifier": _loss(1.5, 0.8),
    # Units fixed 2026-07-11: the model returns mean-reduced per-PIXEL
    # recon+KL, so the old 220 "per image" bar passed anything (observed band
    # 0.066-0.097 — ~3000x under it; gate audit). Collapse detector only: a
    # mean-image decoder scores ~0.05-0.06, inside the band, so this bar
    # can't catch that — sample quality is judged visually in the showcase.
    "vae":                   _loss(0.12, 0.08),
    "unet_ae":               _loss(0.08, 0.03),
    "tabular_diffusion":     _loss(1.0, 0.6),
    # M 0.25 was a pre-data guess. Observed honest band across 4 independent
    # M runs (m-baseline-1..m-triage-5, 5-10 epochs): 0.167/0.180/0.184/0.31 —
    # the bar sat mid-band and flapped. Set below the band floor; L stays
    # ambitious for the full budget.
    "diffusion":             _judge(0.12, 0.50),
    # s_mode=finite: adversarial losses oscillate at equilibrium by design —
    # the downward-trend check misfired on a healthy M run (m-baseline-1).
    # Quality is still gated by judge_score at M/L. M 0.15 was a pre-data
    # guess: the vanilla mini-GAN under the strict confidence x coverage judge
    # sits at 0.031-0.062 with generator EMA across m-triage-5..m-registry-2
    # (0.023-0.139 without EMA, non-monotone). Bar below the observed EMA
    # floor; the coverage term is what keeps it low (partial mode coverage is
    # THE textbook vanilla-GAN failure this item teaches).
    "gan":                   _judge(0.025, 0.40, loss_keys=("g_loss", "d_loss"), s_mode="finite"),
    # Raised 0.10 -> 0.5 (2026-07-11 audit): honest band 0.71-0.77 across 3
    # post-fix runs — the old bar was a stale pre-data guess 7x below it.
    # Caveat in the audit: judges are overconfident on binary noise, so this
    # number flatters pixelcnn vs the continuous samplers; still a real bar.
    "pixelcnn":              _judge(0.5, 0.6),
    "rl_maze":               EvalSpec(metric="success_rate", thresholds={"M": 0.5, "L": 0.8},
                                      loss_keys=("episode_reward", "reward", "loss"), s_mode="finite"),
    "reinforce":             EvalSpec(metric="success_rate", thresholds={"M": 0.4, "L": 0.7},
                                      loss_keys=("episode_reward", "reward", "loss"), s_mode="finite"),
    # ------------------------------------------------------ compositions
    # metric=None: S-style checks (finite + trend + inference probe) at every
    # tier. Numeric metrics land in Phase 2 where each composition exposes one.
    "clip_guided_diffusion":        _composition(loss_keys=("clip_loss", "diff_loss", "loss")),
    "transformer_clip_diffusion":   _composition(loss_keys=("lm_loss", "clip_loss", "diff_loss", "loss")),
    "gan_diffusion_comparison":     _composition(loss_keys=("gan_d_loss", "gan_g_loss", "diff_loss", "loss")),
    # adversarial losses → finite-only S-check (same rationale as "gan")
    "clip_guided_gan":              _composition(loss_keys=("g_loss", "d_loss", "loss"), s_mode="finite"),
    "classifier_guided_diffusion":  _composition(loss_keys=("cls_loss", "diff_loss")),
    "rag_guided_generation":        _composition(),
    "lora_lm":                      _composition(),
    "segment_then_detect":          _composition(loss_keys=("seg_loss", "det_loss", "loss")),
    "multitask_vision":             _composition(),
    "diffusion_distillation":       _composition(loss_keys=("teacher_loss", "student_loss", "loss")),
    "audio_text_contrastive":       _composition(),
    "tabular_text_cross_attention": _composition(),
    "audio_text_dual_encoder":      _composition(),
    "tabular_text_dual_encoder":    _composition(),
    # adversarial losses → finite-only S-check (same rationale as "gan")
    "classifier_guided_gan":        _composition(loss_keys=("g_loss", "d_loss", "loss"), s_mode="finite"),
    "rag_conditioned_diffusion":    _composition(),
    "image_captioning":             _composition(),
    "multimodal_fusion_baseline":   _composition(),
    "latent_diffusion":             _composition(loss_keys=("vae_loss", "latent_loss", "loss")),
}


def get_eval_spec(name: str) -> EvalSpec:
    try:
        return EVAL_SPECS[name]
    except KeyError:
        raise KeyError(
            f"No EvalSpec for {name!r}. Every model/composition must have an entry "
            f"in core/evalspec.py — add one (with threshold justification)."
        ) from None


def passes_threshold(value: float, threshold: float, higher_is_better: bool) -> bool:
    return value >= threshold if higher_is_better else value <= threshold
