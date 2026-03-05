"""CLIP-guided diffusion composition.

Connects three already-trained models:

  CLIPModel       — encodes text queries and digit images into a shared embedding space
  ConditionedUNet — denoises images conditioned on class label + timestep (CFG)
  NoiseScheduler  — DDPM forward/reverse process
  VAE (optional)  — runs diffusion in a compact latent space instead of pixel space

Pipeline
--------
Training (two independent phases):
  1. Train CLIP on MNIST image-text pairs  →  clip.pt
  2. Train ConditionedUNet on MNIST with CFG  →  unet.pt
  (Optional) Pre-train VAE  →  vae.pt

Inference:
  text_to_image(query):
    1. Encode query with CLIP text encoder
    2. Compare against stored per-class text embeddings → nearest class id
    3. Run CFG sampling with that class id (pixel-space or VAE latent)
    4. Decode VAE if used

The rotation trick (dual_oscillation):
  During the reverse diffusion loop, every `flip_every` steps:
    - Rotate the current latent/image 180° with torch.rot90(k=2)
    - Toggle the conditioning class between class_a and class_b
  Effect: the diffusion trajectory oscillates between two classes
  while spatial structure is preserved through the rotation symmetry.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.models.clip.data import MNISTImageTextDataset, label_to_tokens, label_to_all_tokens
from mini_networks.models.clip.model import CLIPModel
from mini_networks.models.diffusion.model import ConditionedUNet
from mini_networks.models.diffusion.scheduler import NoiseScheduler
from mini_networks.models.diffusion.vae import VAE, vae_loss


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class CLIPGuidedDiffusionConfig(BaseConfig):
    model_name: str = "clip_guided_diffusion"

    # CLIP
    embed_dim: int = 128
    vocab_size: int = 256
    text_seq_len: int = 32
    text_d_model: int = 64
    text_n_heads: int = 2
    text_n_layers: int = 2
    clip_temperature: float = 0.07

    # Conditioned diffusion
    n_classes: int = 10
    n_feat: int = 64
    timesteps: int = 400
    beta_start: float = 1e-4
    beta_end: float = 0.02
    drop_prob: float = 0.1        # CFG label-drop probability during training
    guide_weight: float = 2.0    # w in eps = (1+w)*eps_cond - w*eps_uncond

    # VAE (latent diffusion)
    use_vae: bool = False
    vae_latent_channels: int = 4
    vae_kl_weight: float = 1e-3
    vae_pretrain_epochs: int = 5

    # Rotation trick
    flip_every: int = 50          # toggle class + rotate latent every N denoising steps

    dataset: str = "mnist"


# ---------------------------------------------------------------------------
# Composition class
# ---------------------------------------------------------------------------

class CLIPGuidedDiffusion:
    """
    Orchestrates CLIP + ConditionedUNet (+ optional VAE) into a single
    text-to-image pipeline with the rotation oscillation trick.
    """

    def __init__(self):
        self.clip: CLIPModel | None = None
        self.unet: ConditionedUNet | None = None
        self.scheduler: NoiseScheduler | None = None
        self.vae: VAE | None = None
        # Per-class text embeddings stored after CLIP training for retrieval
        self._class_text_embeds: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_clip(self, config: CLIPGuidedDiffusionConfig) -> CLIPModel:
        return CLIPModel(
            embed_dim=config.embed_dim,
            vocab_size=config.vocab_size,
            text_d_model=config.text_d_model,
            text_n_heads=config.text_n_heads,
            text_n_layers=config.text_n_layers,
            text_seq_len=config.text_seq_len,
            temperature=config.clip_temperature,
        ).to(config.device)

    def _build_unet(self, config: CLIPGuidedDiffusionConfig) -> ConditionedUNet:
        in_ch = config.vae_latent_channels if config.use_vae else 1
        return ConditionedUNet(
            in_channels=in_ch,
            n_feat=config.n_feat,
            n_classes=config.n_classes,
            drop_prob=config.drop_prob,
        ).to(config.device)

    def _build_scheduler(self, config: CLIPGuidedDiffusionConfig) -> NoiseScheduler:
        return NoiseScheduler(
            timesteps=config.timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        ).to(torch.device(config.device))

    def _build_vae(self, config: CLIPGuidedDiffusionConfig) -> VAE:
        return VAE(latent_channels=config.vae_latent_channels).to(config.device)

    # ------------------------------------------------------------------
    # Phase 1 — CLIP training
    # ------------------------------------------------------------------

    def train_clip(
        self,
        config: CLIPGuidedDiffusionConfig,
        logger: Logger,
    ) -> None:
        """Train CLIP on MNIST image-text pairs, then cache class embeddings."""
        clip = self._build_clip(config)
        opt = torch.optim.AdamW(clip.parameters(), lr=config.learning_rate)
        ds = MNISTImageTextDataset(
            data_root=config.data_root,
            train=True,
            seq_len=config.text_seq_len,
            vocab_size=config.vocab_size,
            fast_demo=config.fast_demo,
        )
        dl = DataLoader(ds, batch_size=config.effective_batch_size, shuffle=True, num_workers=0)

        for epoch in range(config.effective_epochs):
            clip.train()
            total = 0.0
            for images, tokens, _ in dl:
                images, tokens = images.to(config.device), tokens.to(config.device)
                img_emb, txt_emb = clip(images, tokens)
                loss = clip.contrastive_loss(img_emb, txt_emb)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            avg = total / max(1, len(dl))
            logger.log_metrics(epoch, {"clip_loss": avg})
            print(f"  [CLIP] epoch {epoch}  loss {avg:.4f}")

        self.clip = clip
        torch.save(clip.state_dict(), logger.artifact_path("clip.pt"))
        self._cache_class_embeddings(config)

    def _cache_class_embeddings(self, config: CLIPGuidedDiffusionConfig) -> None:
        """Store one text embedding per digit class by averaging over all caption templates."""
        assert self.clip is not None
        self.clip.eval()
        self._class_text_embeds = {}
        with torch.no_grad():
            for label in range(config.n_classes):
                # Encode every caption in the pool and average → richer class representation
                all_tokens = label_to_all_tokens(label, config.text_seq_len, config.vocab_size)
                all_tokens = all_tokens.to(config.device)           # [N, T]
                embs = self.clip.encode_text(all_tokens)            # [N, D] normalised
                emb = F.normalize(embs.mean(dim=0), dim=-1)         # re-normalise mean
                self._class_text_embeds[label] = emb.cpu()

    # ------------------------------------------------------------------
    # Phase 1b — Optional VAE pre-training
    # ------------------------------------------------------------------

    def train_vae(
        self,
        config: CLIPGuidedDiffusionConfig,
        logger: Logger,
    ) -> None:
        """Pre-train the VAE on MNIST reconstructions."""
        vae = self._build_vae(config)
        opt = torch.optim.Adam(vae.parameters(), lr=config.learning_rate)
        dl = get_dataloader(
            config.dataset, config.data_root, split="train",
            task="classification",
            batch_size=config.effective_batch_size,
            fast_demo=config.fast_demo,
        )
        epochs = 1 if config.fast_demo else config.vae_pretrain_epochs
        for epoch in range(epochs):
            vae.train()
            total = 0.0
            for images, _ in dl:
                images = images.to(config.device) * 2.0 - 1.0   # [-1, 1]
                recon, mu, logvar = vae(images)
                loss = vae_loss(recon, images, mu, logvar, config.vae_kl_weight)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            avg = total / max(1, len(dl))
            logger.log_metrics(epoch, {"vae_loss": avg})
            print(f"  [VAE]  epoch {epoch}  loss {avg:.4f}")
        self.vae = vae
        torch.save(vae.state_dict(), logger.artifact_path("vae.pt"))

    # ------------------------------------------------------------------
    # Phase 2 — Conditioned diffusion training
    # ------------------------------------------------------------------

    def train_diffusion(
        self,
        config: CLIPGuidedDiffusionConfig,
        logger: Logger,
    ) -> None:
        """Train the class-conditioned UNet with classifier-free guidance."""
        unet = self._build_unet(config)
        scheduler = self._build_scheduler(config)
        self.unet = unet
        self.scheduler = scheduler
        opt = torch.optim.Adam(unet.parameters(), lr=config.learning_rate)

        dl = get_dataloader(
            config.dataset, config.data_root, split="train",
            task="classification",
            batch_size=config.effective_batch_size,
            fast_demo=config.fast_demo,
        )

        T = config.timesteps
        for epoch in range(config.effective_epochs):
            unet.train()
            total = 0.0
            for images, labels in dl:
                images = images.to(config.device) * 2.0 - 1.0   # [-1, 1]
                labels = labels.to(config.device)
                B = images.shape[0]

                # Encode to VAE latent if using LDM
                if config.use_vae and self.vae is not None:
                    with torch.no_grad():
                        mu, logvar = self.vae.encode(images)
                        x0 = self.vae.reparameterise(mu, logvar)
                else:
                    x0 = images

                t = torch.randint(0, T, (B,), device=config.device)
                noise = torch.randn_like(x0)
                xt = scheduler.add_noise(x0, noise, t)

                # Randomly drop class labels for CFG training
                context_mask = torch.bernoulli(
                    torch.full((B,), config.drop_prob, device=config.device)
                ).long()

                pred = unet(xt, t, labels, context_mask)
                loss = F.mse_loss(pred, noise)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()

            avg = total / max(1, len(dl))
            logger.log_metrics(epoch, {"diffusion_loss": avg})
            print(f"  [Diffusion] epoch {epoch}  loss {avg:.4f}")

        torch.save(unet.state_dict(), logger.artifact_path("unet.pt"))

    # ------------------------------------------------------------------
    # Convenience: train everything
    # ------------------------------------------------------------------

    def train_all(
        self,
        config: CLIPGuidedDiffusionConfig,
        logger: Logger,
    ) -> None:
        logger.log_config(config.model_dump())
        self.train_clip(config, logger)
        if config.use_vae:
            self.train_vae(config, logger)
        self.train_diffusion(config, logger)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def text_to_class(
        self,
        text_query: str,
        config: CLIPGuidedDiffusionConfig,
    ) -> int:
        """Return the digit class (0-9) whose text embedding best matches the query."""
        assert self.clip is not None, "Train or load CLIP first."
        assert self._class_text_embeds, "Call _cache_class_embeddings() first."
        self.clip.eval()
        tokens = label_to_tokens(0, config.text_seq_len, config.vocab_size)
        # Encode the raw query string character-by-character
        raw = [ord(c) % config.vocab_size for c in text_query]
        raw = raw[:config.text_seq_len] + [0] * max(0, config.text_seq_len - len(raw))
        tokens = torch.tensor(raw, dtype=torch.long).unsqueeze(0).to(config.device)
        with torch.no_grad():
            query_emb = self.clip.encode_text(tokens).squeeze(0).cpu()

        best_class, best_score = 0, -float("inf")
        for cls_id, cls_emb in self._class_text_embeds.items():
            score = (query_emb * cls_emb).sum().item()
            if score > best_score:
                best_score, best_class = score, cls_id
        return best_class

    @torch.no_grad()
    def sample(
        self,
        class_id: int,
        n_samples: int,
        config: CLIPGuidedDiffusionConfig,
    ) -> torch.Tensor:
        """
        Classifier-free guidance sampling.
        eps = (1 + w) * eps_cond  -  w * eps_uncond
        """
        assert self.unet is not None and self.scheduler is not None
        unet, scheduler = self.unet, self.scheduler
        unet.eval()
        dev = config.device

        if config.use_vae and self.vae is not None:
            shape = (n_samples, *self.vae.latent_size)
        else:
            shape = (n_samples, 1, 28, 28)

        x = torch.randn(shape, device=dev)
        labels = torch.full((n_samples,), class_id, dtype=torch.long, device=dev)
        uncond_mask = torch.ones(n_samples, dtype=torch.long, device=dev)   # all unconditional
        cond_mask = torch.zeros(n_samples, dtype=torch.long, device=dev)    # all conditional

        for t in reversed(range(config.timesteps)):
            t_batch = torch.full((n_samples,), t, device=dev, dtype=torch.long)
            eps_cond = unet(x, t_batch, labels, cond_mask)
            eps_uncond = unet(x, t_batch, labels, uncond_mask)
            eps = (1 + config.guide_weight) * eps_cond - config.guide_weight * eps_uncond
            x = scheduler.step(eps, t, x)

        if config.use_vae and self.vae is not None:
            x = self.vae.decode(x)

        return (x.clamp(-1, 1) + 1) / 2   # → [0, 1]

    @torch.no_grad()
    def text_to_image(
        self,
        text_query: str,
        config: CLIPGuidedDiffusionConfig,
        n_samples: int = 4,
    ) -> tuple[torch.Tensor, int]:
        """Full pipeline: text → nearest class → CFG samples. Returns (images, class_id)."""
        class_id = self.text_to_class(text_query, config)
        images = self.sample(class_id, n_samples, config)
        return images, class_id

    @torch.no_grad()
    def dual_oscillation(
        self,
        class_a: int,
        class_b: int,
        config: CLIPGuidedDiffusionConfig,
        flip_every: int | None = None,
        return_frames: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Rotation trick from the legacy reference:
          Every `flip_every` denoising steps:
            1. Rotate the latent 180° with torch.rot90(k=2, dims=(-2,-1))
            2. Toggle conditioning class between class_a and class_b

        This causes the generation trajectory to oscillate between two digit
        classes, producing blended or transitional images.

        Args:
            class_a, class_b: the two digit classes to oscillate between
            flip_every:       toggle interval in denoising steps (default: config.flip_every)
            return_frames:    if True, also return intermediate decoded frames

        Returns:
            final image tensor [1, 1, 28, 28] in [0, 1]
            (and list of frame tensors if return_frames=True)
        """
        assert self.unet is not None and self.scheduler is not None
        unet, scheduler = self.unet, self.scheduler
        unet.eval()
        dev = config.device
        every = flip_every if flip_every is not None else config.flip_every

        if config.use_vae and self.vae is not None:
            shape = (1, *self.vae.latent_size)
        else:
            shape = (1, 1, 28, 28)

        x = torch.randn(shape, device=dev)
        current_class = class_a
        frames: list[torch.Tensor] = []

        uncond_mask = torch.ones(1, dtype=torch.long, device=dev)
        cond_mask = torch.zeros(1, dtype=torch.long, device=dev)

        T = config.timesteps
        for step, t in enumerate(reversed(range(T))):
            label = torch.tensor([current_class], dtype=torch.long, device=dev)
            t_batch = torch.full((1,), t, device=dev, dtype=torch.long)

            eps_cond = unet(x, t_batch, label, cond_mask)
            eps_uncond = unet(x, t_batch, label, uncond_mask)
            eps = (1 + config.guide_weight) * eps_cond - config.guide_weight * eps_uncond
            x = scheduler.step(eps, t, x)

            # ── Rotation trick ──────────────────────────────────────────
            if every > 0 and step > 0 and step % every == 0:
                x = torch.rot90(x, k=2, dims=(-2, -1))   # 180° spatial rotation
                current_class = class_b if current_class == class_a else class_a
            # ────────────────────────────────────────────────────────────

            if return_frames and (step % max(1, T // 16) == 0):
                decoded = self.vae.decode(x) if (config.use_vae and self.vae) else x
                frames.append(((decoded.clamp(-1, 1) + 1) / 2).cpu())

        if config.use_vae and self.vae is not None:
            x = self.vae.decode(x)
        final = (x.clamp(-1, 1) + 1) / 2

        if return_frames:
            return final.cpu(), frames
        return final.cpu()
