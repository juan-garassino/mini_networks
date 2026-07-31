"""The zoo's family tree: atoms, molecules, and compositions.

Every model is classified along two axes:

  introduces  — the atomic MECHANISM(s) this model contributes to the zoo.
                An atom is introduced by exactly one model (unit-enforced);
                its `home` is the file where the mechanism's code lives.
  builds_on   — the zoo models whose mechanisms this model composes.
                ELEMENTARY models build on nothing (the 16 atoms of the
                curriculum); DERIVED models are molecules made of atoms.
                Tokenizer/infra imports (rnn/mamba borrowing CharTokenizer)
                do NOT count as mechanism edges.

Compositions are one level up again: they compose whole MODELS into
pipelines (guided diffusion, latent diffusion, the VLM, ...).

This module is pure data + stdlib helpers (no torch) so catalogs, the CLI,
the /web API, and docs can all read the same source of truth. Sync tests in
tests/test_taxonomy.py enforce full coverage against the registry.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Mechanism:
    description: str
    home: str  # repo-relative path to the file that implements it


@dataclass(frozen=True)
class ModelTaxon:
    level: str                                  # "elementary" | "derived"
    introduces: tuple[str, ...] = ()            # mechanism names (may be empty for derived)
    builds_on: tuple[str, ...] = ()             # parent model names (empty iff elementary)
    note: str = ""                              # one-line "what it adds"


SRC = "src/mini_networks"

MECHANISMS: dict[str, Mechanism] = {
    # ------------------------------------------------ atoms (elementary homes)
    "convolution":               Mechanism("weight-shared local filters + pooling", f"{SRC}/models/classifier/model.py"),
    "mlp":                       Mechanism("fully-connected feature learning", f"{SRC}/models/tabular/model.py"),
    "recurrence":                Mechanism("sequential hidden-state carry (RNN/LSTM/GRU)", f"{SRC}/models/rnn/model.py"),
    "self-attention":            Mechanism("content-based token mixing with a causal mask", f"{SRC}/models/transformer/model.py"),
    "selective-scan":            Mechanism("gated exponential-decay state-space scan", f"{SRC}/models/mamba/model.py"),
    "message-passing":           Mechanism("neighborhood aggregation A_hat X W on a graph", f"{SRC}/models/gnn/model.py"),
    "variational-latent":        Mechanism("stochastic bottleneck trained on the ELBO", f"{SRC}/models/vae/model.py"),
    "adversarial-game":          Mechanism("generator vs discriminator minimax training", f"{SRC}/models/gan/model.py"),
    "iterative-denoising":       Mechanism("forward noising + learned reverse chain (DDPM)", f"{SRC}/models/diffusion/model.py"),
    "autoregressive-factorization": Mechanism("pixel-by-pixel chain rule with masked convs", f"{SRC}/models/pixelcnn/model.py"),
    "skip-encoder-decoder":      Mechanism("U-shaped encoder-decoder with skip connections", f"{SRC}/models/segmentation/unet.py"),
    "contrastive-infonce":       Mechanism("pull augmented views together, push others apart", f"{SRC}/models/simclr/model.py"),
    "policy-gradient":           Mechanism("REINFORCE: reward-weighted log-prob ascent", f"{SRC}/models/reinforce/trainer.py"),
    "value-based-rl":            Mechanism("Q-learning / DQN with an environment loop", f"{SRC}/models/rl_maze/agents.py"),
    "neural-field-rendering":    Mechanism("coordinate MLP + differentiable volume rendering", f"{SRC}/models/nerf/model.py"),
    "tree-search-self-play":     Mechanism("PUCT MCTS distilled into a policy/value net", f"{SRC}/models/alphazero/model.py"),
    # ------------------------------------------- mechanisms added by derived models
    "residual-connection":       Mechanism("identity shortcut around a block", f"{SRC}/models/resnet/model.py"),
    "patch-embedding":           Mechanism("image as a sequence of patch tokens", f"{SRC}/models/vit/model.py"),
    "depthwise-separable-conv":  Mechanism("per-channel conv + 1x1 pointwise mix", f"{SRC}/core/blocks/cnn.py"),
    "modern-conv-recipe":        Mechanism("large-kernel depthwise + inverted bottleneck + LN", f"{SRC}/models/convnext/model.py"),
    "soft-pathway-priors":       Mechanism("constrained+free pathways with per-path L2 priors", f"{SRC}/models/rpp_classifier/model.py"),
    "metric-embedding":          Mechanism("L2-normalized embedding space for retrieval", f"{SRC}/models/vision_embed/model.py"),
    "ema-self-distillation":     Mechanism("EMA teacher + centering/sharpening (DINO)", f"{SRC}/models/dino/model.py"),
    "cross-modal-contrastive":   Mechanism("image and text encoders aligned by InfoNCE (CLIP)", f"{SRC}/models/clip/model.py"),
    "low-rank-adaptation":       Mechanism("frozen base + trainable A@B deltas (LoRA)", f"{SRC}/models/lora/model.py"),
    "bbox-regression":           Mechanism("classification + box-coordinate regression heads", f"{SRC}/models/detection/model.py"),
    "reconstruction-objective":  Mechanism("autoencoding: reproduce the input through a bottleneck", f"{SRC}/models/unet_ae/model.py"),
    "waveform-conv":             Mechanism("1D convolution directly on raw audio", f"{SRC}/models/audio/model.py"),
    "spectrogram-frontend":      Mechanism("STFT magnitude image as a 2D conv input", f"{SRC}/models/audio/model.py"),
    "mel-frontend":              Mechanism("perceptual mel-scaled spectrogram front-end", f"{SRC}/models/audio/model.py"),
    "masked-text-denoising":     Mechanism("mask-ratio diffusion + iterative unmasking (LLaDA)", f"{SRC}/models/text_diffusion/model.py"),
    "encoder-decoder-cross-attention": Mechanism("decoder attends over a separately encoded source", f"{SRC}/models/text_seq2seq/model.py"),
    "bidirectional-encoding":    Mechanism("attention without a causal mask (BERT-style)", f"{SRC}/models/text_token_classifier/model.py"),
    "expert-routing":            Mechanism("top-k router dispatching tokens to expert FFNs", f"{SRC}/models/transformer/model.py"),
    "delta-rule-attention":      Mechanism("linear-time delta-rule recurrence with bounded decay (KDA)", f"{SRC}/models/kimi/model.py"),
    "attention-residuals":       Mechanism("attention over depth: layers read previous block outputs", f"{SRC}/models/kimi/model.py"),
    "latent-moe":                Mechanism("routed experts operating in a compressed latent space", f"{SRC}/models/kimi/model.py"),
    "softcapped-glu":            Mechanism("SiTU-GLU: both GLU factors tanh-softcapped", f"{SRC}/models/kimi/model.py"),
    "kv-compression-attention":  Mechanism("pool m tokens into one KV entry; sparse/dense hybrid (CSA/HCA)", f"{SRC}/models/deepseek/model.py"),
    "hyper-connections":         Mechanism("widened residual stream mixed by doubly-stochastic maps (mHC)", f"{SRC}/models/deepseek/model.py"),
    "multi-token-prediction":    Mechanism("auxiliary head predicting token t+2 (MTP)", f"{SRC}/models/deepseek/model.py"),
    "delayed-generalization":    Mechanism("grokking: val accuracy jumps long after memorization", f"{SRC}/models/grokking/model.py"),
    "retrieval-augmentation":    Mechanism("retriever prepends evidence to the generator's context", f"{SRC}/models/rag/model.py"),
    "reward-model-ppo":          Mechanism("learned reward + PPO fine-tuning of an LM", f"{SRC}/models/rlhf/model.py"),
    "group-relative-advantage":  Mechanism("GRPO: advantages normalized within a sample group", f"{SRC}/models/grpo/trainer.py"),
    "direct-preference-loss":    Mechanism("DPO: preference pairs without an explicit reward model", f"{SRC}/models/dpo/trainer.py"),
    "promptable-decoding":       Mechanism("clicks/boxes select WHICH mask to produce (SAM)", f"{SRC}/models/sam/model.py"),
    "two-way-attention":         Mechanism("prompt tokens and image tokens attend to each other", f"{SRC}/models/sam/model.py"),
}


def _e(introduces: tuple[str, ...], note: str = "") -> ModelTaxon:
    return ModelTaxon(level="elementary", introduces=introduces, note=note)


def _d(builds_on: tuple[str, ...], introduces: tuple[str, ...] = (), note: str = "") -> ModelTaxon:
    return ModelTaxon(level="derived", introduces=introduces, builds_on=builds_on, note=note)


MODEL_TAXONOMY: dict[str, ModelTaxon] = {
    # -------------------------------------------------------- 16 elementary atoms
    "classifier":            _e(("convolution",)),
    "tabular_classifier":    _e(("mlp",)),
    "rnn":                   _e(("recurrence",)),
    "transformer":           _e(("self-attention",)),
    "mamba":                 _e(("selective-scan",)),
    "gnn":                   _e(("message-passing",)),
    "vae":                   _e(("variational-latent",)),
    "gan":                   _e(("adversarial-game",)),
    "diffusion":             _e(("iterative-denoising",)),
    "pixelcnn":              _e(("autoregressive-factorization",)),
    "segmentation":          _e(("skip-encoder-decoder",)),
    "simclr":                _e(("contrastive-infonce",)),
    "reinforce":             _e(("policy-gradient",)),
    "rl_maze":               _e(("value-based-rl",)),
    "nerf":                  _e(("neural-field-rendering",)),
    "alphazero":             _e(("tree-search-self-play",)),
    # ------------------------------------------------------------- 28 derived
    "resnet":                _d(("classifier",), ("residual-connection",), "deeper via identity shortcuts"),
    "vit":                   _d(("transformer", "classifier"), ("patch-embedding",), "attention applied to image patches"),
    "mobilenet":             _d(("classifier",), ("depthwise-separable-conv",), "conv factorized for efficiency"),
    "convnext":              _d(("classifier", "resnet"), ("modern-conv-recipe",), "convs modernized with transformer-era recipes"),
    "rpp_classifier":        _d(("classifier", "tabular_classifier"), ("soft-pathway-priors",), "equivariance as a prior, not a constraint"),
    "vision_embed":          _d(("simclr",), ("metric-embedding",), "contrastive encoder as a retrieval embedding"),
    "dino":                  _d(("vit", "simclr"), ("ema-self-distillation",), "labels replaced by an EMA teacher"),
    "clip":                  _d(("simclr", "transformer"), ("cross-modal-contrastive",), "two modalities, one embedding space"),
    "lora":                  _d(("classifier",), ("low-rank-adaptation",), "adapt a frozen base with tiny deltas"),
    "detection":             _d(("classifier",), ("bbox-regression",), "conv features + localization heads"),
    "unet_ae":               _d(("segmentation",), ("reconstruction-objective",), "SegUNet reused as an autoencoder"),
    "audio_classifier":      _d(("classifier",), ("waveform-conv",), "convolution moved to 1D waveforms"),
    "audio_spectrogram":     _d(("audio_classifier",), ("spectrogram-frontend",), "sound as an image"),
    "audio_melspectrogram":  _d(("audio_spectrogram",), ("mel-frontend",), "perceptual frequency scaling"),
    "audio_transformer":     _d(("transformer", "audio_spectrogram"), (), "attention over spectrogram frames"),
    "tabular_diffusion":     _d(("diffusion", "tabular_classifier"), (), "denoising chain on feature rows"),
    "text_diffusion":        _d(("transformer", "diffusion"), ("masked-text-denoising",), "same transformer, two generation orders"),
    "text_seq2seq":          _d(("transformer",), ("encoder-decoder-cross-attention",), "translate, don't just continue"),
    "text_token_classifier": _d(("transformer",), ("bidirectional-encoding",), "per-token labels, no causal mask"),
    "moe":                   _d(("transformer",), ("expert-routing",), "sparse capacity via a router"),
    "kimi":                  _d(("transformer", "moe", "mamba"),
                                ("delta-rule-attention", "attention-residuals", "latent-moe", "softcapped-glu"),
                                "mini Kimi K3: hybrid linear/global attention frontier LM"),
    "deepseek":              _d(("transformer", "moe"),
                                ("kv-compression-attention", "hyper-connections", "multi-token-prediction"),
                                "mini DeepSeek V4: compressed-KV frontier LM"),
    "grokking":              _d(("transformer",), ("delayed-generalization",), "the phenomenon, not a new architecture"),
    "rag":                   _d(("transformer",), ("retrieval-augmentation",), "retrieve, then generate"),
    "rlhf":                  _d(("transformer", "reinforce"), ("reward-model-ppo",), "align an LM with learned reward"),
    "grpo":                  _d(("rlhf",), ("group-relative-advantage",), "critic-free group advantages"),
    "dpo":                   _d(("rlhf",), ("direct-preference-loss",), "preferences without a reward model"),
    "sam":                   _d(("segmentation", "transformer"), ("promptable-decoding", "two-way-attention"),
                                "the prompt chooses what to segment"),
}

# composition name -> models it composes (whole-model reuse, one level above molecules)
COMPOSITION_TAXONOMY: dict[str, tuple[str, ...]] = {
    "clip_guided_diffusion":        ("clip", "diffusion", "vae"),
    "transformer_clip_diffusion":   ("transformer", "clip", "diffusion"),
    "gan_diffusion_comparison":     ("gan", "diffusion"),
    "clip_guided_gan":              ("clip", "gan"),
    "classifier_guided_diffusion":  ("classifier", "diffusion"),
    "rag_guided_generation":        ("rag", "transformer"),
    "lora_lm":                      ("transformer", "lora"),
    "segment_then_detect":          ("segmentation", "detection"),
    "multitask_vision":             ("classifier", "segmentation", "detection"),
    "diffusion_distillation":       ("diffusion",),
    "audio_text_contrastive":       ("audio_classifier", "transformer"),
    "tabular_text_cross_attention": ("tabular_classifier", "transformer"),
    "audio_text_dual_encoder":      ("audio_classifier", "transformer"),
    "tabular_text_dual_encoder":    ("tabular_classifier", "transformer"),
    "classifier_guided_gan":        ("classifier", "gan"),
    "rag_conditioned_diffusion":    ("rag", "diffusion"),
    "image_captioning":             ("clip", "transformer"),
    "multimodal_fusion_baseline":   ("clip", "classifier"),
    "latent_diffusion":             ("vae", "diffusion"),
    "mode_connect":                 ("classifier",),
    "double_descent":               ("classifier",),
}


def atoms() -> list[str]:
    return [n for n, t in MODEL_TAXONOMY.items() if t.level == "elementary"]


def molecules() -> list[str]:
    return [n for n, t in MODEL_TAXONOMY.items() if t.level == "derived"]


def dependency_edges() -> list[tuple[str, str]]:
    """(child, parent) edges: model→model builds_on plus composition→model."""
    edges = [(name, parent) for name, t in MODEL_TAXONOMY.items() for parent in t.builds_on]
    edges += [(name, m) for name, models in COMPOSITION_TAXONOMY.items() for m in models]
    return edges


def mermaid() -> str:
    """Mermaid flowchart of the model DAG (compositions omitted for legibility)."""
    lines = ["flowchart TD"]
    for name in atoms():
        mech = MODEL_TAXONOMY[name].introduces[0]
        lines.append(f'    {name}[["{name}<br/><i>{mech}</i>"]]')
    for name in molecules():
        lines.append(f'    {name}("{name}")')
    for name, t in MODEL_TAXONOMY.items():
        for parent in t.builds_on:
            lines.append(f"    {parent} --> {name}")
    return "\n".join(lines)
