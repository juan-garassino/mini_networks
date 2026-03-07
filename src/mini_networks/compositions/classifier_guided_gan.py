"""Classifier-guided GAN composition."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from mini_networks.core.config import BaseConfig
from mini_networks.core.data.registry import get_dataloader
from mini_networks.core.logging.logger import Logger
from mini_networks.models.classifier.model import SmallCNN
from mini_networks.models.gan.model import Generator, Discriminator, gan_d_loss, gan_g_loss


class ClassifierGuidedGANConfig(BaseConfig):
    model_name: str = "classifier_guided_gan"
    latent_dim: int = 100
    num_classes: int = 10
    cls_hidden: int = 64
    cls_weight: float = 1.0
    dataset: str = "mnist"


class ClassifierGuidedGAN:
    def __init__(self):
        self.G: Generator | None = None
        self.D: Discriminator | None = None
        self.C: SmallCNN | None = None

    def _build_classifier(self, config: ClassifierGuidedGANConfig) -> SmallCNN:
        return SmallCNN(hidden_dim=config.cls_hidden, num_classes=config.num_classes).to(config.device)

    def train(self, config: ClassifierGuidedGANConfig, logger: Logger) -> None:
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
        C = self._build_classifier(config)
        self.G, self.D, self.C = G, D, C

        opt_g = torch.optim.Adam(G.parameters(), lr=config.learning_rate)
        opt_d = torch.optim.Adam(D.parameters(), lr=config.learning_rate)
        opt_c = torch.optim.Adam(C.parameters(), lr=config.learning_rate)
        bce = torch.nn.BCELoss()

        for epoch in range(config.effective_epochs):
            total_g = 0.0
            total_d = 0.0
            total_c = 0.0
            for images, labels in dl:
                images, labels = images.to(config.device), labels.to(config.device)
                images = images * 2.0 - 1.0
                B = images.size(0)

                # Train classifier on real images
                logits = C((images + 1.0) / 2.0)
                c_loss = F.cross_entropy(logits, labels)
                opt_c.zero_grad()
                c_loss.backward()
                opt_c.step()
                total_c += c_loss.item()

                # Train D
                z = torch.randn(B, config.latent_dim, device=config.device)
                fake = G(z)
                d_loss = gan_d_loss(D, images, fake, bce)
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                total_d += d_loss.item()

                # Train G with classifier guidance
                z = torch.randn(B, config.latent_dim, device=config.device)
                fake = G(z)
                g_loss = gan_g_loss(D, fake, bce)
                cls_logits = C((fake + 1.0) / 2.0)
                cls_loss = F.cross_entropy(cls_logits, labels)
                loss = g_loss + config.cls_weight * cls_loss
                opt_g.zero_grad()
                loss.backward()
                opt_g.step()
                total_g += g_loss.item()

            n = max(1, len(dl))
            logger.log_metrics(epoch, {
                "g_loss": total_g / n,
                "d_loss": total_d / n,
                "c_loss": total_c / n,
            })

        torch.save(G.state_dict(), logger.artifact_path("generator.pt"))
        torch.save(D.state_dict(), logger.artifact_path("discriminator.pt"))
        torch.save(C.state_dict(), logger.artifact_path("classifier.pt"))
