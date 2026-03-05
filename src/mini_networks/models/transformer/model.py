"""Decoder-only TransformerLM with pluggable FFN blocks (standard / MoE / Mamba)."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FFN blocks — each forward() returns (output, aux_loss)
# ---------------------------------------------------------------------------

class StandardFFN(nn.Module):
    """Two-layer FFN: Linear → GELU → Linear → Dropout."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x), torch.tensor(0.0, device=x.device)


class _TopKRouter(nn.Module):
    """Gumbel-softmax top-k router for MoEFFN."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_experts: int,
        k: int = 1,
        add_gumbel: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )
        self.k = k
        self.add_gumbel = add_gumbel
        self.temperature = float(temperature)

    @staticmethod
    def _gumbel_noise(shape, device):
        u = torch.rand(shape, device=device).clamp_(1e-9, 1 - 1e-9)
        return -torch.log(-torch.log(u))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (sparse_probs [N, E], dense_probs [N, E])."""
        logits = self.net(x)
        if self.add_gumbel and self.training:
            logits = logits + self._gumbel_noise(logits.shape, logits.device)
        probs = F.softmax(logits / self.temperature, dim=-1)
        if self.k >= probs.size(-1):
            return probs, probs
        topk_vals, topk_idx = torch.topk(probs, self.k, dim=-1)
        mask = torch.zeros_like(probs).scatter_(-1, topk_idx, torch.ones_like(topk_vals))
        sp = probs * mask
        sp = sp / (sp.sum(dim=-1, keepdim=True) + 1e-9)
        return sp, probs


class MoEFFN(nn.Module):
    """
    Mixture-of-Experts FFN with:
      - shared bottom/top projection (scaled by a learnable shared_scale)
      - N expert up/down projections (only top-k active per token)
      - Gumbel-softmax TopK router
      - balance loss (KL-to-uniform) + entropy bonus

    forward() returns (output, aux_loss).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 4,
        k: int = 1,
        router_hidden: int = 64,
        dropout: float = 0.1,
        balance_loss_weight: float = 0.02,
        entropy_bonus_weight: float = 0.001,
        shared_scale: float = 0.3,
        add_gumbel: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.blw = balance_loss_weight
        self.entw = entropy_bonus_weight

        # Shared path (always active)
        self.shared_fc = nn.Linear(d_model, d_ff)
        self.shared_proj = nn.Linear(d_ff, d_model)
        self.shared_scale = nn.Parameter(torch.tensor(float(shared_scale)))

        # Per-expert paths (only top-k activate per token)
        self.expert_fc = nn.ModuleList([nn.Linear(d_model, d_ff) for _ in range(num_experts)])
        self.expert_proj = nn.ModuleList([nn.Linear(d_ff, d_model) for _ in range(num_experts)])

        self.router = _TopKRouter(d_model, router_hidden, num_experts, k, add_gumbel, temperature)
        self.route_norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, T, C]. Returns (output [B, T, C], aux_loss scalar)."""
        B, T, C = x.shape
        xf = x.view(B * T, C)

        # Shared path
        shared = self.shared_proj(self.act(self.shared_fc(xf))) * self.shared_scale

        # Routing
        sp, dp = self.router(self.route_norm(xf))  # [BT, E], [BT, E]

        # Weighted expert sum
        routed = torch.zeros(B * T, C, device=x.device)
        for e in range(self.num_experts):
            h = self.expert_proj[e](self.act(self.expert_fc[e](xf)))
            routed = routed + sp[:, e].unsqueeze(-1) * h

        y = (shared + routed).view(B, T, C)
        y = self.drop(y)

        # Balance loss: KL(mean routing probs || uniform) − entropy bonus
        eps = 1e-9
        E = self.num_experts
        m = dp.mean(dim=0)
        kl = torch.sum(m * (m.clamp(eps).log() - math.log(1.0 / E)))
        ent = -(dp.clamp(eps) * dp.clamp(eps).log()).sum(dim=-1).mean()
        aux = self.blw * kl - self.entw * ent

        return y, aux


class MambaFFN(nn.Module):
    """
    Dependency-free SSM-style block:
      - depthwise Conv1d for local mixing
      - gated exponential decay scan (simplified Mamba selective state space)

    Residual connection is applied internally.
    forward() returns (output, aux_loss=0.0).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj_in = nn.Linear(d_model, 2 * d_model)
        self.dwconv = nn.Conv1d(
            2 * d_model, 2 * d_model,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=2 * d_model,
        )
        self.act = nn.SiLU()
        self.a = nn.Parameter(torch.zeros(d_model))  # decay log-rate
        self.b = nn.Parameter(torch.zeros(d_model))  # input projection
        self.proj_out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, T, C]. Returns (output [B, T, C], aux_loss=0.0)."""
        B, T, C = x.shape
        h = self.norm(x)
        h = self.proj_in(h)               # [B, T, 2C]
        h = h.transpose(1, 2)             # [B, 2C, T]
        h = self.dwconv(h)[:, :, :T]      # causal trim → [B, 2C, T]
        h = h.transpose(1, 2)             # [B, T, 2C]

        u, g = h.chunk(2, dim=-1)         # each [B, T, C]
        g = torch.sigmoid(g)
        u = self.act(u)

        # Gated exponential decay SSM scan
        a = torch.exp(-F.softplus(self.a))   # [C], decay ∈ (0, 1)
        b = self.b                            # [C], input scale
        ys = []
        s = torch.zeros(B, C, device=x.device)
        for t in range(T):
            s = a * s + b * u[:, t, :]
            ys.append(s)
        y = torch.stack(ys, dim=1)       # [B, T, C]
        y = g * y
        y = self.drop(self.proj_out(y))
        return x + y, torch.tensor(0.0, device=x.device)


# ---------------------------------------------------------------------------
# Block factory
# ---------------------------------------------------------------------------

def _make_ffn(block_type: str, d_model: int, d_ff: int, dropout: float, **kwargs) -> nn.Module:
    if block_type == "moe":
        return MoEFFN(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=kwargs.get("num_experts", 4),
            k=kwargs.get("k", 1),
            router_hidden=kwargs.get("router_hidden", 64),
            dropout=dropout,
            balance_loss_weight=kwargs.get("balance_loss_weight", 0.02),
            entropy_bonus_weight=kwargs.get("entropy_bonus_weight", 0.001),
            shared_scale=kwargs.get("shared_scale", 0.3),
            add_gumbel=kwargs.get("add_gumbel", True),
            temperature=kwargs.get("temperature", 1.0),
        )
    elif block_type == "mamba":
        return MambaFFN(
            d_model=d_model,
            d_state=kwargs.get("d_state", 16),
            d_conv=kwargs.get("d_conv", 4),
            dropout=dropout,
        )
    else:
        return StandardFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)


# ---------------------------------------------------------------------------
# Transformer block and LM
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        block_type: str = "standard",
        **ffn_kwargs,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = _make_ffn(block_type, d_model, d_ff, dropout, **ffn_kwargs)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x, aux_loss). aux_loss is 0 for non-MoE blocks."""
        T = x.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, is_causal=True)
        x = self.norm1(x + self.drop(attn_out))
        ffn_out, aux = self.ffn(x)
        x = self.norm2(x + self.drop(ffn_out))
        return x, aux


class TransformerLM(nn.Module):
    """Decoder-only language model with pluggable FFN blocks."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        seq_len: int = 128,
        dropout: float = 0.1,
        block_type: str = "standard",
        **ffn_kwargs,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, block_type, **ffn_kwargs)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.seq_len = seq_len

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits [B, T, V], total_aux_loss scalar)."""
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.token_embed(tokens) + self.pos_embed(positions)
        total_aux = torch.tensor(0.0, device=tokens.device)
        for block in self.blocks:
            x, aux = block(x)
            total_aux = total_aux + aux
        x = self.norm(x)
        return self.lm_head(x), total_aux

    @torch.no_grad()
    def generate(
        self, prompt: torch.Tensor, max_new_tokens: int = 64, temperature: float = 1.0
    ) -> torch.Tensor:
        self.eval()
        x = prompt
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.seq_len:]
            logits, _ = self(x_cond)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)
        return x
