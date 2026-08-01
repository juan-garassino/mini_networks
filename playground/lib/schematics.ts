// Simplified anatomy of every species — a tiny DSL rendered by
// components/network-flow.tsx as a 3b1b-style animated forward pass.
// k: node kind · n: node count · l: label. skip=true draws a residual arc
// from this stage to the next-but-one; loop=true draws a recurrence arc.
export type StageKind =
  | "in" | "conv" | "dense" | "attn" | "recur" | "latent"
  | "router" | "noise" | "out";
export interface Stage {
  k: StageKind; n?: number; l?: string; skip?: boolean; loop?: boolean;
}

const LM: Stage[] = [
  { k: "in", n: 5, l: "chars" }, { k: "attn", n: 5, l: "attention" },
  { k: "dense", n: 4, l: "ffn" }, { k: "out", n: 5, l: "next char" },
];
const CNN: Stage[] = [
  { k: "in", n: 4, l: "28×28" }, { k: "conv", n: 4, l: "conv" },
  { k: "conv", n: 3, l: "conv" }, { k: "dense", n: 3, l: "fc" },
  { k: "out", n: 3, l: "10 logits" },
];

export const SCHEMATICS: Record<string, Stage[]> = {
  // atoms
  classifier: CNN,
  tabular_classifier: [
    { k: "in", n: 4, l: "features" }, { k: "dense", n: 5, l: "hidden" },
    { k: "dense", n: 4 }, { k: "out", n: 3, l: "classes" }],
  rnn: [
    { k: "in", n: 4, l: "chars" }, { k: "recur", n: 3, l: "hidden state", loop: true },
    { k: "out", n: 4, l: "next char" }],
  transformer: LM,
  mamba: [
    { k: "in", n: 5, l: "chars" }, { k: "conv", n: 4, l: "dw conv" },
    { k: "recur", n: 4, l: "decay scan", loop: true }, { k: "out", n: 5 }],
  gnn: [
    { k: "in", n: 5, l: "nodes" }, { k: "attn", n: 5, l: "Â · message pass" },
    { k: "attn", n: 5, l: "2nd hop" }, { k: "out", n: 4, l: "communities" }],
  vae: [
    { k: "in", n: 4, l: "image" }, { k: "conv", n: 3, l: "encode" },
    { k: "latent", n: 2, l: "z ~ N(μ,σ)" }, { k: "conv", n: 3, l: "decode" },
    { k: "out", n: 4, l: "recon" }],
  gan: [
    { k: "latent", n: 2, l: "z" }, { k: "conv", n: 3, l: "generator" },
    { k: "in", n: 4, l: "fake | real" }, { k: "conv", n: 3, l: "discriminator" },
    { k: "out", n: 1, l: "real?" }],
  diffusion: [
    { k: "noise", n: 4, l: "noise" }, { k: "conv", n: 4, l: "unet ↺ T steps", loop: true },
    { k: "out", n: 4, l: "image" }],
  pixelcnn: [
    { k: "in", n: 5, l: "pixels so far" }, { k: "conv", n: 4, l: "masked conv" },
    { k: "out", n: 1, l: "next pixel" }],
  segmentation: [
    { k: "in", n: 4, l: "image" }, { k: "conv", n: 3, l: "down", skip: true },
    { k: "latent", n: 2, l: "bottleneck" }, { k: "conv", n: 3, l: "up" },
    { k: "out", n: 4, l: "mask" }],
  simclr: [
    { k: "in", n: 4, l: "two views" }, { k: "conv", n: 3, l: "encoder ×2" },
    { k: "dense", n: 3, l: "project" }, { k: "latent", n: 2, l: "pull / push" }],
  reinforce: [
    { k: "in", n: 3, l: "state" }, { k: "dense", n: 4, l: "policy" },
    { k: "out", n: 4, l: "action ~ π" }, { k: "recur", n: 1, l: "reward → ∇", loop: true }],
  rl_maze: [
    { k: "in", n: 3, l: "state" }, { k: "dense", n: 4, l: "Q(s,·)" },
    { k: "out", n: 4, l: "argmax a" }, { k: "recur", n: 1, l: "bootstrap", loop: true }],
  nerf: [
    { k: "in", n: 3, l: "(x,y,z)" }, { k: "dense", n: 5, l: "PE + mlp" },
    { k: "latent", n: 2, l: "σ, rgb" }, { k: "out", n: 4, l: "∫ render" }],
  alphazero: [
    { k: "in", n: 3, l: "board" }, { k: "attn", n: 4, l: "MCTS tree", loop: true },
    { k: "dense", n: 3, l: "policy·value" }, { k: "out", n: 3, l: "move" }],
  // compounds (terse variations)
  resnet: [
    { k: "in", n: 4 }, { k: "conv", n: 4, l: "block", skip: true },
    { k: "conv", n: 4, l: "+ identity" }, { k: "out", n: 3 }],
  rpp_classifier: [
    { k: "in", n: 4 }, { k: "conv", n: 3, l: "conv path" },
    { k: "dense", n: 3, l: "+ free path" }, { k: "out", n: 3 }],
  unet_ae: SCHEMATICS_PLACEHOLDER(),
  mobilenet: [
    { k: "in", n: 4 }, { k: "conv", n: 4, l: "depthwise" },
    { k: "conv", n: 3, l: "1×1 mix" }, { k: "out", n: 3 }],
  detection: [
    { k: "in", n: 4 }, { k: "conv", n: 4, l: "backbone" },
    { k: "dense", n: 2, l: "cls | bbox" }, { k: "out", n: 2, l: "class + box" }],
  sam: [
    { k: "in", n: 4, l: "image + click" }, { k: "attn", n: 4, l: "two-way attn" },
    { k: "conv", n: 3, l: "upsample" }, { k: "out", n: 3, l: "3 masks" }],
  convnext: [
    { k: "in", n: 4 }, { k: "conv", n: 4, l: "7×7 dw", skip: true },
    { k: "dense", n: 4, l: "inverted mlp" }, { k: "out", n: 3 }],
  lora: [
    { k: "in", n: 4 }, { k: "conv", n: 4, l: "frozen base" },
    { k: "latent", n: 2, l: "+ A·B (rank r)" }, { k: "out", n: 3 }],
  vit: [
    { k: "in", n: 4, l: "patches" }, { k: "attn", n: 4, l: "encoder" },
    { k: "dense", n: 3, l: "cls token" }, { k: "out", n: 3 }],
  audio_classifier: [
    { k: "in", n: 5, l: "waveform" }, { k: "conv", n: 4, l: "1d conv" },
    { k: "out", n: 3, l: "digit" }],
  audio_spectrogram: [
    { k: "in", n: 5, l: "stft image" }, { k: "conv", n: 4 }, { k: "out", n: 3 }],
  audio_melspectrogram: [
    { k: "in", n: 5, l: "mel image" }, { k: "conv", n: 4 }, { k: "out", n: 3 }],
  audio_transformer: [
    { k: "in", n: 5, l: "spec frames" }, { k: "attn", n: 4 }, { k: "out", n: 3 }],
  text_token_classifier: [
    { k: "in", n: 5, l: "tokens" }, { k: "attn", n: 5, l: "no causal mask" },
    { k: "out", n: 5, l: "label / token" }],
  moe: [
    { k: "in", n: 5 }, { k: "attn", n: 5 }, { k: "router", n: 4, l: "top-k router" },
    { k: "out", n: 5 }],
  text_seq2seq: [
    { k: "in", n: 4, l: "source" }, { k: "attn", n: 4, l: "encode" },
    { k: "attn", n: 4, l: "cross-attend" }, { k: "out", n: 4, l: "target" }],
  kimi: [
    { k: "in", n: 5 }, { k: "recur", n: 4, l: "KDA ×3", loop: true },
    { k: "attn", n: 4, l: "gated attn" }, { k: "router", n: 4, l: "latent moe" },
    { k: "out", n: 5 }],
  deepseek: [
    { k: "in", n: 5 }, { k: "latent", n: 3, l: "compress m→1" },
    { k: "attn", n: 4, l: "sparse attend" }, { k: "router", n: 4, l: "moe" },
    { k: "out", n: 5 }],
  grokking: [
    { k: "in", n: 3, l: "a ∘ b" }, { k: "attn", n: 4, l: "2 layers" },
    { k: "out", n: 3, l: "mod 97" }],
  rag: [
    { k: "in", n: 3, l: "query" }, { k: "latent", n: 3, l: "retrieve" },
    { k: "attn", n: 4, l: "generate" }, { k: "out", n: 4 }],
  text_diffusion: [
    { k: "noise", n: 5, l: "masked" }, { k: "attn", n: 5, l: "bidirectional ↺", loop: true },
    { k: "out", n: 5, l: "unmasked" }],
  tabular_diffusion: [
    { k: "noise", n: 4, l: "noisy row" }, { k: "dense", n: 4, l: "mlp ↺", loop: true },
    { k: "out", n: 4, l: "row" }],
  vision_embed: [
    { k: "in", n: 4 }, { k: "conv", n: 3 }, { k: "latent", n: 2, l: "unit vector" }],
  clip: [
    { k: "in", n: 4, l: "image | text" }, { k: "conv", n: 3, l: "two encoders" },
    { k: "latent", n: 2, l: "shared space" }],
  dino: [
    { k: "in", n: 4, l: "views" }, { k: "attn", n: 4, l: "student | ema teacher" },
    { k: "latent", n: 2, l: "match" }],
  rlhf: [
    { k: "in", n: 4 }, { k: "attn", n: 4, l: "LM" }, { k: "dense", n: 2, l: "reward" },
    { k: "recur", n: 1, l: "ppo", loop: true }],
  grpo: [
    { k: "in", n: 4 }, { k: "attn", n: 4, l: "LM ×group" },
    { k: "dense", n: 3, l: "group advantage" }, { k: "recur", n: 1, loop: true }],
  dpo: [
    { k: "in", n: 4, l: "pair" }, { k: "attn", n: 4, l: "policy | ref" },
    { k: "out", n: 1, l: "preference loss" }],
};

function SCHEMATICS_PLACEHOLDER(): Stage[] {
  return [
    { k: "in", n: 4 }, { k: "conv", n: 3, l: "down", skip: true },
    { k: "latent", n: 2 }, { k: "conv", n: 3, l: "up" },
    { k: "out", n: 4, l: "recon" }];
}

export function schematicFor(name: string, composes?: string[]): Stage[] {
  if (SCHEMATICS[name]) return SCHEMATICS[name];
  if (composes?.length)
    return [
      ...composes.map((m) => ({ k: "conv" as const, n: 3, l: m })),
      { k: "out", n: 3, l: name },
    ];
  return LM;
}
