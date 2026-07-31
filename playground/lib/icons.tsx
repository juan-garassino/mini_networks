// One distinct Lucide glyph per species — stroke icons, drawn like the rest
// of the blueprint's pen line-work. Atoms get literal mechanism icons;
// compounds get their "twist"; reactions get apparatus/process glyphs.
import type { LucideIcon } from "lucide-react";
import {
  Activity, Anchor, ArrowLeftRight, AudioWaveform, BarChart3, Blend, Bone,
  BookOpen, Camera, CloudFog, Combine, Compass, CornerUpRight, Crosshair,
  Columns2, Dices, Eraser, Eye, Feather, Fingerprint, FlaskConical, FlipVertical2,
  GitBranch, Grid3x3, Grip, Group, Hash, Headphones, Highlighter,
  Gem, IterationCw, Lasso, Layers, LayoutGrid, Library, Lightbulb, Link2, Map,
  Magnet, MessageSquareText, Mic, Milestone, MousePointerClick, Music2,
  Navigation, Orbit, PenTool, Plug, Puzzle, Radio, Rocket, Scale, ScanSearch,
  Target,
  ScrollText, Share2, Shrink, Spline, Split, Swords, Table2,
  TableProperties, Users, Wand2, Waves, Workflow,
} from "lucide-react";

export const MODEL_ICONS: Record<string, LucideIcon> = {
  // ————— atoms
  classifier: Grid3x3,            // conv: a sliding filter grid
  tabular_classifier: Table2,     // rows in, labels out
  rnn: IterationCw,               // the recurrence loop
  transformer: Eye,               // attention looks
  mamba: Waves,                   // the scan flows
  gnn: Share2,                    // message passing
  vae: Shrink,                    // the bottleneck
  gan: Swords,                    // the adversarial duel
  diffusion: CloudFog,            // noise clearing
  pixelcnn: Grip,                 // pixel by pixel
  segmentation: Lasso,            // trace the mask
  simclr: Magnet,                 // pull views together
  reinforce: Dices,               // stochastic policy
  rl_maze: Map,                   // navigate for value
  nerf: Orbit,                    // cameras on orbit
  alphazero: Hash,                // the tic-tac-toe board itself
  // ————— compounds
  resnet: CornerUpRight,          // the skip connection
  rpp_classifier: Split,          // two pathways
  unet_ae: FlipVertical2,         // mirror reconstruction
  mobilenet: Feather,             // lightweight
  detection: ScanSearch,          // find the box
  sam: MousePointerClick,         // the prompt click
  convnext: Wand2,                // convs, modernized
  lora: Puzzle,                   // a small adapter piece
  vit: LayoutGrid,                // patches as tokens
  audio_classifier: AudioWaveform,
  audio_spectrogram: BarChart3,   // frequency bins
  audio_melspectrogram: Music2,   // perceptual scale
  text_token_classifier: Highlighter,
  moe: Users,                     // the experts
  audio_transformer: Radio,
  text_seq2seq: GitBranch,        // encode, then branch to decode
  kimi: Rocket,                   // frontier LM
  deepseek: Anchor,               // the deep one
  rag: BookOpen,                  // retrieve first
  grokking: Lightbulb,            // the sudden click
  text_diffusion: Eraser,         // unmask by rounds
  tabular_diffusion: TableProperties,
  vision_embed: Fingerprint,      // an identity vector
  clip: Link2,                    // two modalities linked
  dino: Bone,                     // self-distillation (and, well, dino)
  rlhf: Scale,                    // judged by reward
  grpo: Group,                    // group-relative
  dpo: Milestone,                 // straight to preference
};

export const REACTION_ICONS: Record<string, LucideIcon> = {
  clip_guided_diffusion: Compass,
  transformer_clip_diffusion: PenTool,
  gan_diffusion_comparison: ArrowLeftRight,
  clip_guided_gan: Crosshair,
  classifier_guided_diffusion: Navigation,
  rag_guided_generation: Library,
  lora_lm: Plug,
  segment_then_detect: Layers,
  multitask_vision: Workflow,
  diffusion_distillation: FlaskConical,
  audio_text_contrastive: Mic,
  tabular_text_cross_attention: Combine,
  audio_text_dual_encoder: Headphones,
  tabular_text_dual_encoder: Columns2,
  classifier_guided_gan: Target,
  rag_conditioned_diffusion: ScrollText,
  image_captioning: MessageSquareText,
  multimodal_fusion_baseline: Blend,
  latent_diffusion: Gem,
  mode_connect: Spline,
  double_descent: Activity,
  vlm: Camera,
};

export function iconFor(name: string): LucideIcon | null {
  return MODEL_ICONS[name] ?? REACTION_ICONS[name] ?? null;
}
