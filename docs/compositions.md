# Compositions

Compositions are small, explicit adapters that connect multiple models through simple interfaces. Each composition is a teaching artifact and not a black box.

**Shared Composition Bases**
- `CompositionBase` enforces a consistent `infer(config, inputs)` signature.
- `ContrastiveCompositionBase` provides a shared contrastive training loop for paired modalities.
- Diffusion compositions use a shared sampling driver so guidance, logging, and scheduling are consistent.

**Available Compositions**
- `clip_guided_diffusion` uses CLIP similarity to guide diffusion sampling.
- `transformer_clip_diffusion` generates prompts with a Transformer, selects a class with CLIP, and samples with diffusion.
- `gan_diffusion_comparison` compares GAN vs diffusion side‑by‑side.
- `clip_guided_gan` adds CLIP similarity to GAN generator loss.
- `classifier_guided_diffusion` uses a classifier’s gradients to guide diffusion.
- `rag_guided_generation` retrieves context before LM generation.
- `lora_lm` LoRA adapter fine‑tuning for the TransformerLM.
- `segment_then_detect` uses segmentation masks to infer bounding boxes.
- `multitask_vision` shared encoder with classification + segmentation + detection heads.
- `diffusion_distillation` distills a diffusion teacher into a small denoiser.
- `audio_text_contrastive` audio-text contrastive alignment.
- `tabular_text_cross_attention` tabular-text cross-attention alignment.
- `audio_text_dual_encoder` audio-text dual-encoder contrastive.
- `tabular_text_dual_encoder` tabular-text dual-encoder contrastive.
- `classifier_guided_gan` classifier-guided GAN.
- `rag_conditioned_diffusion` RAG-conditioned diffusion.
- `image_captioning` image-to-text captioning.
- `multimodal_fusion_baseline` image+text fusion classifier.
- `latent_diffusion` VAE + latent diffusion.

**Standard Inference Inputs**
All composition inference calls now expect a dict under `inputs`. Common keys:
- Audio-text (contrastive or dual-encoder): `{"waves": ..., "labels": ...}`
- Tabular-text (contrastive or dual-encoder): `{"features": ..., "labels": ...}`

**Guiding Interfaces**
Models that participate in compositions should expose small, predictable hooks:
- `encode()` for embedding inputs.
- `score()` for similarity or ranking.
- `guided_step()` for guidance integration.
- `sample()` for generation.

These hooks keep composition code short and make the learning goal explicit.

**API Usage**
Compositions are exposed under the `/compose` API prefix:
- `POST /compose/{composition_name}` starts training.
- `POST /compose/{composition_name}/infer` runs inference.
