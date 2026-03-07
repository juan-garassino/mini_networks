"""CLIP-guided GAN composition: add CLIP similarity to generator loss."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.models.clip.model import CLIPModel
from mini_networks.models.clip.data import label_to_tokens
from mini_networks.models.gan.model import Generator, Discriminator, gan_d_loss, gan_g_loss


class CLIPGuidedGANConfig(BaseConfig):
    model_name: str = "clip_guided_gan"

    # GAN
    latent_dim: int = 100
    clip_weight: float = 1.0

    # CLIP
    embed_dim: int = 64
    vocab_size: int = 256
    text_seq_len: int = 32
    text_d_model: int = 32
    text_n_heads: int = 2
    text_n_layers: int = 1
    clip_temperature: float = 0.07

    dataset: str = "mnist"


class CLIPGuidedGAN:
    def __init__(self):
        self.G: Generator | None = None
        self.D: Discriminator | None = None
        self.clip: CLIPModel | None = None

    def _build_clip(self, config: CLIPGuidedGANConfig) -> CLIPModel:
        return CLIPModel(
            embed_dim=config.embed_dim,
            vocab_size=config.vocab_size,
            text_d_model=config.text_d_model,
            text_n_heads=config.text_n_heads,
            text_n_layers=config.text_n_layers,
            text_seq_len=config.text_seq_len,
            temperature=config.clip_temperature,
        ).to(config.device)

    def train(self, config: CLIPGuidedGANConfig, logger: Logger) -> None:
        dl = get_dataloader(
            name=config.dataset,
            data_root=config.data_root,
            split="train",
            task="classification",
            batch_size=config.effective_batch_size,
            fast_demo=config.fast_demo,
        )

        G = Generator(latent_dim=config.latent_dim).to(config.device)
        D = Discriminator().to(config.device)
        clip = self._build_clip(config)
        self.G, self.D, self.clip = G, D, clip

        opt_g = torch.optim.Adam(G.parameters(), lr=config.learning_rate)
        opt_d = torch.optim.Adam(D.parameters(), lr=config.learning_rate)
        opt_c = torch.optim.Adam(clip.parameters(), lr=config.learning_rate)
        bce = torch.nn.BCELoss()

        for epoch in range(config.effective_epochs):
            total_g = 0.0
            total_d = 0.0
            total_clip = 0.0
            for images, labels in dl:
                images, labels = images.to(config.device), labels.to(config.device)
                images = images * 2.0 - 1.0
                B = images.size(0)

                # Train CLIP on real images + labels
                tokens = torch.stack([
                    label_to_tokens(int(l), config.text_seq_len, config.vocab_size)
                    for l in labels
                ], dim=0).to(config.device)
                images_clip = (images + 1.0) / 2.0
                img_emb, txt_emb = clip(images_clip, tokens)
                clip_loss = clip.contrastive_loss(img_emb, txt_emb)
                opt_c.zero_grad()
                clip_loss.backward()
                opt_c.step()
                total_clip += clip_loss.item()

                # Train D
                z = torch.randn(B, config.latent_dim, device=config.device)
                fake = G(z)
                d_loss = gan_d_loss(D, images, fake, bce)
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                total_d += d_loss.item()

                # Train G with CLIP guidance
                z = torch.randn(B, config.latent_dim, device=config.device)
                fake = G(z)
                g_loss = gan_g_loss(D, fake, bce)
                fake_img_emb, fake_txt_emb = clip((fake + 1.0) / 2.0, tokens)
                cos = F.cosine_similarity(fake_img_emb, fake_txt_emb, dim=-1).mean()
                clip_guidance = 1.0 - cos
                loss = g_loss + config.clip_weight * clip_guidance
                opt_g.zero_grad()
                loss.backward()
                opt_g.step()
                total_g += g_loss.item()

            n = max(1, len(dl))
            logger.log_metrics(epoch, {
                "g_loss": total_g / n,
                "d_loss": total_d / n,
                "clip_loss": total_clip / n,
            })

        torch.save(G.state_dict(), logger.artifact_path("generator.pt"))
        torch.save(D.state_dict(), logger.artifact_path("discriminator.pt"))
        torch.save(clip.state_dict(), logger.artifact_path("clip.pt"))

    @torch.no_grad()
    def sample(self, config: CLIPGuidedGANConfig, n: int = 4) -> torch.Tensor:
        if self.G is None:
            raise RuntimeError("Train the model first.")
        z = torch.randn(n, config.latent_dim, device=config.device)
        return (self.G(z) + 1.0) / 2.0
