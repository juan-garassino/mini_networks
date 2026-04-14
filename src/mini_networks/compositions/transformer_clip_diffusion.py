"""Transformer → CLIP → Diffusion composition.

Full three-model pipeline
--------------------------
  Stage 1 — TransformerLM is trained on a text corpus.
  Stage 2 — CLIPModel is trained on MNIST image-text pairs.
  Stage 3 — ConditionedUNet is trained on MNIST with classifier-free guidance.

Inference (generate_image)
--------------------------
  1. TransformerLM: generate `k_prompts` candidate text continuations from a seed.
  2. CLIPModel: encode every candidate and the stored per-class embeddings;
     find the digit class whose text embedding best matches the average candidate.
  3. ConditionedUNet (CFG): sample an image conditioned on that class.

Educational value
-----------------
  - Shows how a generative language model can "steer" image generation.
  - Demonstrates cross-modal retrieval (text → class id via cosine similarity).
  - The three models are fully independent; only their outputs are chained.
"""
from __future__ import annotations


import torch
import torch.nn.functional as F

from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.models.clip.data import label_to_all_tokens
from mini_networks.models.clip.model import CLIPModel
from mini_networks.models.diffusion.model import ConditionedUNet
from mini_networks.models.diffusion.scheduler import NoiseScheduler
from mini_networks.core.diffusion.sampling import sample_loop
from mini_networks.models.transformer.model import TransformerLM
from mini_networks.models.transformer.tokenizer import CharTokenizer
from mini_networks.core.config import BaseConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TransformerCLIPDiffusionConfig(BaseConfig):
    model_name: str = "transformer_clip_diffusion"

    # ── Transformer LM ────────────────────────────────────────────────
    lm_d_model: int = 64
    lm_n_heads: int = 2
    lm_n_layers: int = 2
    lm_d_ff: int = 128
    lm_seq_len: int = 64
    lm_dropout: float = 0.1
    lm_epochs: int = 2            # independent from effective_epochs

    # ── CLIP ──────────────────────────────────────────────────────────
    embed_dim: int = 64
    vocab_size: int = 256         # shared char vocab (0-255)
    text_seq_len: int = 32
    text_d_model: int = 32
    text_n_heads: int = 2
    text_n_layers: int = 1
    clip_temperature: float = 0.07
    n_classes: int = 10
    clip_epochs: int = 2

    # ── Conditioned diffusion ─────────────────────────────────────────
    n_feat: int = 32
    timesteps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02
    drop_prob: float = 0.1
    guide_weight: float = 2.0
    diff_epochs: int = 2

    # ── Inference ─────────────────────────────────────────────────────
    k_prompts: int = 8            # number of candidate prompts to generate
    prompt_max_new: int = 16      # tokens generated per candidate
    prompt_temperature: float = 1.0

    dataset: str = "mnist"
    text_file: str = ""


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

class TransformerCLIPDiffusion:
    """Three-stage composition: LM generates text → CLIP ranks → diffusion generates image."""

    def __init__(self):
        self.lm: TransformerLM | None = None
        self.tokenizer: CharTokenizer | None = None
        self.clip: CLIPModel | None = None
        self.unet: ConditionedUNet | None = None
        self.scheduler: NoiseScheduler | None = None
        self._class_embeds: dict[int, torch.Tensor] = {}  # label → normalized text embed

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_lm(self, config: TransformerCLIPDiffusionConfig, vocab_size: int) -> TransformerLM:
        return TransformerLM(
            vocab_size=vocab_size,
            d_model=config.lm_d_model,
            n_heads=config.lm_n_heads,
            n_layers=config.lm_n_layers,
            d_ff=config.lm_d_ff,
            seq_len=config.lm_seq_len,
            dropout=config.lm_dropout,
        ).to(config.device)

    def _build_clip(self, config: TransformerCLIPDiffusionConfig) -> CLIPModel:
        return CLIPModel(
            embed_dim=config.embed_dim,
            vocab_size=config.vocab_size,
            text_d_model=config.text_d_model,
            text_n_heads=config.text_n_heads,
            text_n_layers=config.text_n_layers,
            text_seq_len=config.text_seq_len,
            temperature=config.clip_temperature,
        ).to(config.device)

    def _build_unet(self, config: TransformerCLIPDiffusionConfig) -> ConditionedUNet:
        return ConditionedUNet(
            in_channels=1,
            n_feat=config.n_feat,
            n_classes=config.n_classes,
            drop_prob=config.drop_prob,
        ).to(config.device)

    def _build_scheduler(self, config: TransformerCLIPDiffusionConfig) -> NoiseScheduler:
        return NoiseScheduler(
            timesteps=config.timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        ).to(torch.device(config.device))

    # ------------------------------------------------------------------
    # Stage 1 — TransformerLM
    # ------------------------------------------------------------------

    def train_lm(self, config: TransformerCLIPDiffusionConfig, logger: Logger) -> None:
        """Train a character-level LM on Tiny Shakespeare (or custom text_file)."""
        import torch.nn.functional as F
        import torch.optim as optim

        dl = get_dataloader(
            name="text_file",
            data_root=config.data_root,
            split="train",
            batch_size=config.effective_batch_size,
            fast_demo=config.effective_fast_demo,
            sample_limit=config.dataset_sample_limit,
            file_path=config.text_file,
            seq_len=config.lm_seq_len,
        )
        ds = dl.dataset

        self.tokenizer = ds.tokenizer
        vocab_size = ds.vocab_size

        model = self._build_lm(config, vocab_size)
        self.lm = model
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

        epochs = config.tier_epochs(config.lm_epochs, medium_cap=2)
        for epoch in range(epochs):
            model.train()
            total = 0.0
            for x, y in dl:
                x, y = x.to(config.device), y.to(config.device)
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item()
            avg = total / max(1, len(dl))
            logger.log_metrics(epoch, {"lm_loss": avg})
            print(f"  [LM]   epoch {epoch}  loss {avg:.4f}")

        torch.save(model.state_dict(), logger.artifact_path("lm.pt"))
        self.tokenizer.save(str(logger.artifact_path("tokenizer.json")))

    # ------------------------------------------------------------------
    # Stage 2 — CLIP
    # ------------------------------------------------------------------

    def train_clip(self, config: TransformerCLIPDiffusionConfig, logger: Logger) -> None:
        """Train CLIP on MNIST image-text pairs."""
        import torch.optim as optim

        clip = self._build_clip(config)
        opt = optim.AdamW(clip.parameters(), lr=config.learning_rate)
        dl = get_dataloader(
            name="mnist",
            data_root=config.data_root,
            split="train",
            task="clip",
            batch_size=config.effective_batch_size,
            fast_demo=config.effective_fast_demo,
            sample_limit=config.dataset_sample_limit,
            seq_len=config.text_seq_len,
            vocab_size=config.vocab_size,
        )

        epochs = config.tier_epochs(config.clip_epochs, medium_cap=2)
        for epoch in range(epochs):
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

    def _cache_class_embeddings(self, config: TransformerCLIPDiffusionConfig) -> None:
        """Precompute per-class text embeddings by averaging all caption templates."""
        assert self.clip is not None
        self.clip.eval()
        self._class_embeds = {}
        with torch.no_grad():
            for label in range(config.n_classes):
                all_tokens = label_to_all_tokens(
                    label, config.text_seq_len, config.vocab_size
                ).to(config.device)
                embs = self.clip.encode_text(all_tokens)           # [N, D]
                emb = F.normalize(embs.mean(dim=0), dim=-1)        # [D]
                self._class_embeds[label] = emb.cpu()

    # ------------------------------------------------------------------
    # Stage 3 — Conditioned diffusion
    # ------------------------------------------------------------------

    def train_diffusion(self, config: TransformerCLIPDiffusionConfig, logger: Logger) -> None:
        """Train class-conditioned UNet with CFG."""
        import torch.optim as optim

        unet = self._build_unet(config)
        scheduler = self._build_scheduler(config)
        self.unet = unet
        self.scheduler = scheduler
        opt = optim.Adam(unet.parameters(), lr=config.learning_rate)

        dl = get_dataloader(
            config.dataset, config.data_root, split="train",
            task="classification",
            batch_size=config.effective_batch_size,
            fast_demo=config.effective_fast_demo,
            sample_limit=config.dataset_sample_limit,
        )

        T = config.timesteps
        epochs = config.tier_epochs(config.diff_epochs, medium_cap=2)
        for epoch in range(epochs):
            unet.train()
            total = 0.0
            for images, labels in dl:
                images = images.to(config.device) * 2.0 - 1.0
                labels = labels.to(config.device)
                B = images.shape[0]
                t = torch.randint(0, T, (B,), device=config.device)
                noise = torch.randn_like(images)
                xt = scheduler.add_noise(images, noise, t)
                ctx_mask = torch.bernoulli(
                    torch.full((B,), config.drop_prob, device=config.device)
                ).long()
                pred = unet(xt, t, labels, ctx_mask)
                loss = F.mse_loss(pred, noise)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            avg = total / max(1, len(dl))
            logger.log_metrics(epoch, {"diff_loss": avg})
            print(f"  [Diff] epoch {epoch}  loss {avg:.4f}")

        torch.save(unet.state_dict(), logger.artifact_path("unet.pt"))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def train_all(self, config: TransformerCLIPDiffusionConfig, logger: Logger) -> None:
        logger.log_config(config.model_dump())
        self.train_lm(config, logger)
        self.train_clip(config, logger)
        self.train_diffusion(config, logger)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate_prompts(
        self,
        seed: str,
        config: TransformerCLIPDiffusionConfig,
    ) -> list[str]:
        """Use the LM to generate `k_prompts` candidate text continuations."""
        assert self.lm is not None and self.tokenizer is not None
        model = self.lm
        model.eval()

        ids = self.tokenizer.encode(seed)
        if not ids:
            ids = [0]
        prompt = torch.tensor([ids], dtype=torch.long, device=config.device)

        candidates: list[str] = []
        for _ in range(config.k_prompts):
            with torch.no_grad():
                out = model.generate(
                    prompt,
                    max_new_tokens=config.prompt_max_new,
                    temperature=config.prompt_temperature,
                )
            candidates.append(self.tokenizer.decode(out[0].tolist()))
        return candidates

    def rank_prompts_by_class(
        self,
        prompts: list[str],
        config: TransformerCLIPDiffusionConfig,
    ) -> tuple[int, list[float]]:
        """Return (best_class_id, per-class similarity scores).

        Each prompt is encoded by CLIP; the average embedding is matched
        against stored per-class embeddings to find the closest digit class.
        """
        assert self.clip is not None and self._class_embeds, "Train CLIP first."
        self.clip.eval()

        # Encode all prompts
        prompt_embeds: list[torch.Tensor] = []
        with torch.no_grad():
            for text in prompts:
                raw = [ord(c) % config.vocab_size for c in text]
                raw = raw[:config.text_seq_len] + [0] * max(0, config.text_seq_len - len(raw))
                tokens = torch.tensor(raw, dtype=torch.long, device=config.device).unsqueeze(0)
                emb = self.clip.encode_text(tokens).squeeze(0).cpu()
                prompt_embeds.append(emb)

        # Average embedding across all candidate prompts
        avg_embed = F.normalize(torch.stack(prompt_embeds).mean(dim=0), dim=-1)

        # Cosine similarity against each class
        scores: list[float] = []
        for cls_id in range(config.n_classes):
            cls_emb = self._class_embeds[cls_id]
            scores.append((avg_embed * cls_emb).sum().item())

        best_class = int(torch.tensor(scores).argmax().item())
        return best_class, scores

    @torch.no_grad()
    def sample_class(
        self,
        class_id: int,
        config: TransformerCLIPDiffusionConfig,
        n_samples: int = 4,
    ) -> torch.Tensor:
        """CFG sampling for a given class. Returns [n, 1, 28, 28] in [0, 1]."""
        assert self.unet is not None and self.scheduler is not None
        unet, scheduler = self.unet, self.scheduler
        unet.eval()
        dev = config.device
        labels = torch.full((n_samples,), class_id, dtype=torch.long, device=dev)
        cond = torch.zeros(n_samples, dtype=torch.long, device=dev)
        uncond = torch.ones(n_samples, dtype=torch.long, device=dev)
        x = sample_loop(
            scheduler=scheduler,
            predict_noise=lambda x, t_b, t, _: (
                (1 + config.guide_weight) * unet(x, t_b, labels, cond)
                - config.guide_weight * unet(x, t_b, labels, uncond)
            ),
            shape=(n_samples, 1, 28, 28),
            device=dev,
            timesteps=config.timesteps,
        )
        return ((x.clamp(-1, 1) + 1) / 2).cpu()

    def generate_image(
        self,
        seed: str,
        config: TransformerCLIPDiffusionConfig,
        n_samples: int = 4,
    ) -> tuple[torch.Tensor, int, list[str]]:
        """Full pipeline: seed text → LM prompts → CLIP ranking → diffusion image.

        Returns:
            images [n, 1, 28, 28] in [0, 1]
            predicted class_id (0–9)
            list of generated candidate prompts
        """
        prompts = self.generate_prompts(seed, config)
        class_id, _ = self.rank_prompts_by_class(prompts, config)
        images = self.sample_class(class_id, config, n_samples=n_samples)
        return images, class_id, prompts
