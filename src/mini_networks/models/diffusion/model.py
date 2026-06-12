"""UNet noise-prediction backbones for DDPM: unconditional and class-conditioned.

Key idea: DDPM does not generate images directly — it trains a network
eps_theta(x_t, t) to predict the Gaussian noise that was mixed into a clean image,
with loss ||eps - eps_theta(sqrt(a_bar_t) x_0 + sqrt(1 - a_bar_t) eps, t)||^2.
Sampling then runs that predictor backwards through the scheduler, step by step.
A UNet fits the job: the output must be the same shape as the input, and skip
connections preserve the fine detail the bottleneck throws away.

This implementation, for 28x28 MNIST. UNet (unconditional): sinusoidal embedding
of t → 2-layer MLP; encoder c=32 → 64 → 128 channels with two stride-2 downsamples
(28 → 14 → 7); bottleneck ResBlock + self-attention over the 7x7=49 positions +
ResBlock; decoder mirrors up with channel-concat skips. Each ResBlock injects the
time embedding additively per channel between its two convs. ConditionedUNet (for
classifier-free guidance): pools the encoder to a 1x1 vector, then modulates the
decoder as cemb * features + temb, where class one-hot and normalised t/T each
pass through small MLPs; during training labels are zeroed with prob drop_prob=0.1
so one network learns both conditional and unconditional scores, enabling
eps = (1 + w) * eps_cond - w * eps_uncond at sampling time.

Deliberately simplified vs Ho et al. 2020: tiny channel widths, one attention
block at the bottleneck only, F.interpolate patches up odd spatial sizes instead
of careful padding, and the conditioned variant feeds t as a normalised scalar
(not sinusoidal) and crushes the bottleneck to 1x1, trading spatial detail for
simplicity.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal time embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device).float() / (half - 1)
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)
        h, _ = self.attn(h, h, h)
        return x + h.transpose(1, 2).view(B, C, H, W)


class UNet(nn.Module):
    """Lightweight UNet for DDPM on 28x28 images."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        time_dim: int = 128,
    ):
        super().__init__()
        c = base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(128, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )

        # Encoder
        self.enc0 = nn.Conv2d(in_channels, c, 3, padding=1)
        self.enc1 = ResBlock(c, c * 2, time_dim)
        self.down1 = nn.Conv2d(c * 2, c * 2, 4, stride=2, padding=1)
        self.enc2 = ResBlock(c * 2, c * 4, time_dim)
        self.down2 = nn.Conv2d(c * 4, c * 4, 4, stride=2, padding=1)

        # Bottleneck
        self.mid1 = ResBlock(c * 4, c * 4, time_dim)
        self.mid_attn = SelfAttention(c * 4)
        self.mid2 = ResBlock(c * 4, c * 4, time_dim)

        # Decoder
        self.up2 = nn.ConvTranspose2d(c * 4, c * 4, 4, stride=2, padding=1)
        self.dec2 = ResBlock(c * 8, c * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(c * 2, c * 2, 4, stride=2, padding=1)
        self.dec1 = ResBlock(c * 4, c, time_dim)

        self.out_norm = nn.GroupNorm(min(8, c), c)
        self.out_conv = nn.Conv2d(c, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = get_time_embedding(t, 128)
        t_emb = self.time_mlp(t_emb)

        e0 = self.enc0(x)
        e1 = self.enc1(e0, t_emb)
        d1 = self.down1(e1)
        e2 = self.enc2(d1, t_emb)
        d2 = self.down2(e2)

        m = self.mid1(d2, t_emb)
        m = self.mid_attn(m)
        m = self.mid2(m, t_emb)

        u2 = self.up2(m)
        # Handle size mismatch from stride
        u2 = F.interpolate(u2, size=e2.shape[-2:])
        u2 = self.dec2(torch.cat([u2, e2], dim=1), t_emb)
        u1 = self.up1(u2)
        u1 = F.interpolate(u1, size=e1.shape[-2:])
        u1 = self.dec1(torch.cat([u1, e1], dim=1), t_emb)

        return self.out_conv(F.silu(self.out_norm(u1)))


# ---------------------------------------------------------------------------
# Class-conditioned UNet for classifier-free guidance (CLIP-guided diffusion)
# ---------------------------------------------------------------------------

class EmbedFC(nn.Module):
    """Two-layer MLP projector used for time and class conditioning."""

    def __init__(self, input_dim: int, emb_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(-1, self.input_dim))


class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpBlock(nn.Module):
    """Upsample x, then concatenate with skip connection, then conv."""

    def __init__(self, x_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(x_ch, x_ch, 4, stride=2, padding=1)
        combined = x_ch + skip_ch
        self.conv = nn.Sequential(
            nn.Conv2d(combined, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = F.interpolate(x, size=skip.shape[-2:])
        x = torch.cat([x, skip], dim=1)   # concatenate along channel dim
        return self.conv(x)


class ConditionedUNet(nn.Module):
    """
    Class + time conditioned UNet for classifier-free guidance DDPM.

    Conditioning mechanism (from legacy reference):
      - Time and class labels each projected via EmbedFC to embedding vectors
      - In decoder: features = class_emb * features + time_emb  (multiplicative + additive)
      - Classifier-free guidance: during training, class labels randomly dropped (zeroed)
        with probability `drop_prob` so the model learns both conditional and unconditional

    Args:
        in_channels:  image channels (1 for MNIST)
        n_feat:       base channel width
        n_classes:    number of conditioning classes (10 for digits)
        drop_prob:    probability of dropping class label during training (CFG)
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_feat: int = 64,
        n_classes: int = 10,
        drop_prob: float = 0.1,
    ):
        super().__init__()
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.drop_prob = drop_prob

        # Encoder
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, n_feat, 3, padding=1),
            nn.GroupNorm(min(8, n_feat), n_feat), nn.GELU(),
        )
        self.down1 = _DownBlock(n_feat, n_feat * 2)
        self.down2 = _DownBlock(n_feat * 2, n_feat * 4)
        self.to_vec = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.GELU())

        # Time embeddings (scalar t/T → emb)
        self.time_embed1 = EmbedFC(1, n_feat * 4)
        self.time_embed2 = EmbedFC(1, n_feat * 2)

        # Class embeddings (one-hot → emb)
        self.class_embed1 = EmbedFC(n_classes, n_feat * 4)
        self.class_embed2 = EmbedFC(n_classes, n_feat * 2)

        # Decoder
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(n_feat * 4, n_feat * 4, 7, 7),
            nn.GroupNorm(min(8, n_feat * 4), n_feat * 4), nn.GELU(),
        )
        # x_ch=n_feat*4 (from u0), skip_ch=n_feat*2 (from d1), out=n_feat*2
        self.up1 = _UpBlock(x_ch=n_feat * 4, skip_ch=n_feat * 2, out_ch=n_feat * 2)
        # x_ch=n_feat*2 (from u1), skip_ch=n_feat (from x0), out=n_feat
        self.up2 = _UpBlock(x_ch=n_feat * 2, skip_ch=n_feat,     out_ch=n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, padding=1),
            nn.GroupNorm(min(8, n_feat), n_feat), nn.GELU(),
            nn.Conv2d(n_feat, in_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            noisy image  [B, C, H, W]
            t:            integer timestep [B] (will be normalised to [0,1] internally)
            c:            class labels [B] (long) — None means unconditional
            context_mask: bool/long [B] — 1 = drop this label (force unconditional)
        """
        B = x.shape[0]
        t_scaled = (t.float() / t.float().max().clamp(min=1)).view(B, 1)

        # One-hot class labels; zero out masked (unconditional) examples
        if c is not None:
            c_onehot = F.one_hot(c.long(), num_classes=self.n_classes).float()
            if context_mask is not None:
                c_onehot = c_onehot * (1.0 - context_mask.float().unsqueeze(1))
        else:
            c_onehot = torch.zeros(B, self.n_classes, device=x.device)

        # Encoder
        x0 = self.init_conv(x)                     # [B, n_feat,   H,   W]
        d1 = self.down1(x0)                         # [B, n_feat*2, H/2, W/2]
        d2 = self.down2(d1)                         # [B, n_feat*4, H/4, W/4]
        hidden = self.to_vec(d2)                    # [B, n_feat*4, 1,   1]

        # Condition embeddings → spatial tensors
        cemb1 = self.class_embed1(c_onehot).view(B, self.n_feat * 4, 1, 1)
        temb1 = self.time_embed1(t_scaled).view(B, self.n_feat * 4, 1, 1)
        cemb2 = self.class_embed2(c_onehot).view(B, self.n_feat * 2, 1, 1)
        temb2 = self.time_embed2(t_scaled).view(B, self.n_feat * 2, 1, 1)

        # Decoder with multiplicative class + additive time conditioning
        u0 = self.up0(hidden)                       # [B, n_feat*4, 7,   7]
        u1 = self.up1(cemb1 * u0 + temb1, d1)      # [B, n_feat*2, H/2, W/2]
        u2 = self.up2(cemb2 * u1 + temb2, x0)      # [B, n_feat,   H,   W]

        return self.out(u2)
