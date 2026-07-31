"""Mini SAM: promptable segmentation — the prompt decides WHAT to segment.

Key idea (Segment Anything, arXiv 2304.02643): decouple perception from
intent. One image contains many valid masks; instead of baking one
segmentation task into the weights, condition the mask decoder on a PROMPT
(clicks, boxes) that selects the target. Because a single click is often
ambiguous (part vs whole, this object vs that one), the decoder predicts
THREE candidate masks plus its own IoU estimate for each, and training
backpropagates only the best-matching candidate (min-loss over heads) — so
the heads specialize on different plausible interpretations.

This implementation (defaults): 56x56 two-MNIST-digit composites, where
ambiguity is real — the same image has (at least) two correct masks and the
click picks one. Image encoder = 3-conv net to 14x14x64 tokens + learned 2D
positions; prompt encoder = Fourier features of the (y,x) coordinate + a
learned type embedding (positive / negative / box corners); mask decoder =
2 rounds of two-way attention (prompt+output tokens <-> image tokens), then
each mask token dot-products with 2x-upsampled image features to produce a
56x56 logit map; an IoU head scores the three candidates and inference
returns the self-rated best.

Key equations:
  loss = mean_batch min_head [BCE(mask_h, gt) + Dice(mask_h, gt)]
         + MSE(iou_pred_winner, actual IoU of the winning mask)
  Fourier prompt encoding: [sin(2*pi*B p); cos(2*pi*B p)], B ~ N(0,1) frozen

Deliberately simplified vs the paper: conv encoder instead of a ViT-H, no
text prompts, no mask-input prompts, no iterative click refinement; the
composite-digit task stands in for SA-1B's ambiguity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample soft dice loss. probs/target: [B, H, W] -> [B]."""
    num = 2 * (probs * target).sum(dim=(-2, -1)) + eps
    den = probs.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + eps
    return 1 - num / den


class PromptEncoder(nn.Module):
    """Points/box corners -> prompt tokens via frozen Fourier features + type embeds."""

    TYPES = 4  # 0 positive point, 1 negative point, 2 box top-left, 3 box bottom-right

    def __init__(self, embed_dim: int, n_freq: int = 16):
        super().__init__()
        self.register_buffer("freq", torch.randn(2, n_freq))  # frozen, in state_dict
        self.proj = nn.Linear(2 * n_freq, embed_dim)
        self.type_embed = nn.Embedding(self.TYPES, embed_dim)

    def forward(self, coords: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        """coords [B, P, 2] in [0,1], types [B, P] -> tokens [B, P, C]."""
        ang = 2 * torch.pi * coords @ self.freq          # [B, P, n_freq]
        feats = torch.cat([ang.sin(), ang.cos()], dim=-1)
        return self.proj(feats) + self.type_embed(types)


class TwoWayBlock(nn.Module):
    """Tokens attend to image; image attends back — both directions per round."""

    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        self.t2i = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.i2t = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm_t1 = nn.LayerNorm(embed_dim)
        self.norm_t2 = nn.LayerNorm(embed_dim)
        self.norm_i = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim), nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, tokens: torch.Tensor, image: torch.Tensor):
        a, _ = self.t2i(tokens, image, image)
        tokens = self.norm_t1(tokens + a)
        tokens = self.norm_t2(tokens + self.mlp(tokens))
        b, _ = self.i2t(image, tokens, tokens)
        image = self.norm_i(image + b)
        return tokens, image


class MiniSAM(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_decoder_layers: int = 2,
        n_masks: int = 3,
        canvas: int = 56,
    ):
        super().__init__()
        self.n_masks = n_masks
        self.canvas = canvas
        grid = canvas // 4  # 14x14 tokens after two stride-2 convs

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, embed_dim, 1),
        )
        self.img_pos = nn.Parameter(torch.randn(1, grid * grid, embed_dim) * 0.02)
        self.prompt_encoder = PromptEncoder(embed_dim)
        self.out_tokens = nn.Parameter(torch.randn(n_masks + 1, embed_dim) * 0.02)  # +1 iou token
        self.decoder = nn.ModuleList(
            [TwoWayBlock(embed_dim, n_heads) for _ in range(n_decoder_layers)]
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 32, 2, stride=2), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.GELU(),
        )
        self.mask_mlps = nn.ModuleList(
            [nn.Linear(embed_dim, 16) for _ in range(n_masks)]
        )
        self.iou_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, n_masks),
        )

    def forward(
        self, image: torch.Tensor, coords: torch.Tensor, types: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """image [B,1,H,W], coords [B,P,2] in [0,1], types [B,P]
        -> (mask_logits [B,n_masks,H,W], iou_pred [B,n_masks])."""
        B = image.shape[0]
        feat = self.encoder(image)                            # [B, C, g, g]
        g = feat.shape[-1]
        img_tokens = feat.flatten(2).transpose(1, 2) + self.img_pos  # [B, g*g, C]

        prompt = self.prompt_encoder(coords, types)           # [B, P, C]
        tokens = torch.cat(
            [self.out_tokens.unsqueeze(0).expand(B, -1, -1), prompt], dim=1
        )
        for block in self.decoder:
            tokens, img_tokens = block(tokens, img_tokens)

        feat = img_tokens.transpose(1, 2).view(B, -1, g, g)
        up = self.upsample(feat)                              # [B, 16, H, W]
        masks = torch.stack(
            [
                torch.einsum("bc,bchw->bhw", mlp(tokens[:, i]), up)
                for i, mlp in enumerate(self.mask_mlps)
            ],
            dim=1,
        )                                                     # [B, n_masks, H, W]
        iou_pred = self.iou_head(tokens[:, self.n_masks])     # the iou token
        return masks, iou_pred
