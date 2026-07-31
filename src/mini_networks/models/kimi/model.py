"""Mini Kimi K3: hybrid linear/global attention LM with LatentMoE FFNs.

Key idea (Kimi K3 technical report, arXiv 2607.24653): scale information flow
along three axes at once — sequence (Kimi Delta Attention, a linear-time
delta-rule recurrence, hybridized 3:1 with gated full attention), depth
(Attention Residuals: each layer *attends* over previous block outputs instead
of summing one residual stream), and width (Stable LatentMoE: routed experts
that work in a small latent space, activated sparsely per token).

This implementation (defaults): 2 hybrid blocks of [3 KDA + 1 gated full
attention] + 1 final global layer, d_model=128, every attention layer paired
with a Stable LatentMoE FFN (2 shared + top-2 of 8 routed latent experts,
SiTU-GLU activation, aux-loss-free bias balancing) — except the FIRST layer,
whose FFN is dense (paper Table 1: 1 dense layer). A Multi-Token Prediction
module (Table 1: 1 MTP layer) predicts token t+2 as an auxiliary objective.
No positional embeddings anywhere (NoPE): the KDA short-convs + per-channel
decay carry position, the full-attention layers do pure content addressing.

Key equations:
  KDA recurrence   S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ,
                   o_t = S_tᵀ q_t, with α_t = exp(g_min · σ(e^{A_h} z_t))
                   bounding every retention factor in (e^{g_min}, 1)  (Eq. 5)
  Output gates     y = W_o [σ(W_g x) ⊙ õ]  on both KDA and full attention
  AttnRes          h_l = Σ_i softmax_i(w_lᵀ RMSNorm(b_i)) · b_i over
                   [embedding, previous block sums, running partial sum]
  SiTU-GLU         (β₁ tanh(W_g x/β₁) · σ(W_g x)) ⊙ β₂ tanh(W_u x/β₂),
                   bounding both GLU factors that SwiGLU leaves unbounded

Deliberately simplified vs the paper: the KDA recurrence is a plain Python
loop over time (the chunkwise/Tensor-Core form and 16-token tiling are
throughput artifacts at 1T scale — seq_len≤128 here); plain MHA stands in for
gated MLA (latent-KV compression is DeepSeek's lesson — see models/deepseek);
load balancing uses the classic sign-rule bias update (K3's Quantile
Balancing needs a distributed histogram estimator over million-token batches);
no Muon, no MTP, no vision pathway, no FP8.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_networks.core.blocks.norm import RMSNorm


class _CausalShortConv(nn.Module):
    """Depthwise causal Conv1d over time — the ShortConv in KDA's q/k/v paths."""

    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel, padding=kernel - 1, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        return self.conv(x.transpose(1, 2))[:, :, :T].transpose(1, 2)


class KDALayer(nn.Module):
    """Kimi Delta Attention: delta-rule recurrence with bounded channel-wise decay."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        g_min: float = -5.0,
        conv_kernel: int = 3,
        decay_rank: int = 16,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.g_min = g_min

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.conv_q = _CausalShortConv(d_model, conv_kernel)
        self.conv_k = _CausalShortConv(d_model, conv_kernel)
        self.conv_v = _CausalShortConv(d_model, conv_kernel)

        self.w_beta = nn.Linear(d_model, n_heads)
        # Low-rank decay logits z + per-head-channel bias + per-head log-scale A (Eq. 2/5)
        self.alpha_down = nn.Linear(d_model, decay_rank, bias=False)
        self.alpha_up = nn.Linear(decay_rank, d_model, bias=False)
        self.alpha_bias = nn.Parameter(torch.zeros(d_model))
        self.log_scale = nn.Parameter(torch.zeros(n_heads))  # A_h, init 0

        self.head_norm = RMSNorm(self.d_head)
        self.w_gate = nn.Linear(d_model, d_model)  # full-rank output gate (Eq. 6)
        self.w_o = nn.Linear(d_model, d_model)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.d_head

        q = F.normalize(F.silu(self.conv_q(self.w_q(x))).view(B, T, H, D), dim=-1)
        k = F.normalize(F.silu(self.conv_k(self.w_k(x))).view(B, T, H, D), dim=-1)
        v = self._heads(F.silu(self.conv_v(self.w_v(x))))
        beta = torch.sigmoid(self.w_beta(x))  # [B, T, H]

        # Bounded log-decay: g = g_min * sigmoid(e^{A_h} * z), alpha = exp(g)
        z = self._heads(self.alpha_up(self.alpha_down(x)) + self.alpha_bias)
        g = self.g_min * torch.sigmoid(torch.exp(self.log_scale).view(1, 1, H, 1) * z)
        alpha = torch.exp(g)  # [B, T, H, D] in (e^{g_min}, 1)

        # Delta-rule recurrence: S_t = Diag(alpha) S + beta k (v - kT Diag(alpha) S)T
        S = x.new_zeros(B, H, D, D)  # [B, H, d_k, d_v]
        outs = []
        for t in range(T):
            S = alpha[:, t].unsqueeze(-1) * S
            k_t = k[:, t]                                      # [B, H, D]
            pred = torch.einsum("bhk,bhkv->bhv", k_t, S)       # kT S
            err = beta[:, t].unsqueeze(-1) * (v[:, t] - pred)  # write strength * residual
            S = S + k_t.unsqueeze(-1) * err.unsqueeze(-2)
            outs.append(torch.einsum("bhkv,bhk->bhv", S, q[:, t]))
        o = torch.stack(outs, dim=1)  # [B, T, H, D]

        o = self.head_norm(o).reshape(B, T, C)
        return self.w_o(torch.sigmoid(self.w_gate(x)) * o)


class GatedAttention(nn.Module):
    """Causal full attention with NoPE and a full-rank sigmoid output gate (Eq. 7).

    Stands in for K3's gated MLA: global content addressing between KDA layers;
    the latent KV compression itself is demonstrated in models/deepseek.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.w_gate = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        o, _ = self.attn(x, x, x, attn_mask=mask, is_causal=True)
        return self.w_o(torch.sigmoid(self.w_gate(x)) * o)


class SiTUGLU(nn.Module):
    """SiTU-GLU MLP: softcapped Swish gate ⊙ softcapped up branch → down proj (Eq. 12)."""

    def __init__(self, dim: int, hidden: int, beta1: float = 4.0, beta2: float = 25.0):
        super().__init__()
        self.w_g = nn.Linear(dim, hidden)
        self.w_u = nn.Linear(dim, hidden)
        self.w_d = nn.Linear(hidden, dim)
        self.beta1 = beta1
        self.beta2 = beta2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.w_g(x)
        gate = self.beta1 * torch.tanh(g / self.beta1) * torch.sigmoid(g)
        up = self.beta2 * torch.tanh(self.w_u(x) / self.beta2)
        return self.w_d(gate * up)


class StableLatentMoE(nn.Module):
    """Shared expert(s) + routed experts in latent space, RMSNorm before up-proj (Eq. 11).

    Routing is aux-loss-free: a non-learned per-expert bias steers Top-k
    selection (sign-rule update toward uniform load, frozen at eval) without
    entering the mixture weights p_i.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        latent_dim: int = 64,
        num_routed: int = 8,
        top_k: int = 2,
        num_shared: int = 1,
        beta1: float = 4.0,
        beta2: float = 25.0,
        balance_gamma: float = 0.01,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_routed = num_routed
        self.top_k = top_k
        self.gamma = balance_gamma

        self.shared = nn.ModuleList(
            [SiTUGLU(d_model, d_ff, beta1, beta2) for _ in range(num_shared)]
        )
        self.w_down = nn.Linear(d_model, latent_dim, bias=False)
        self.experts = nn.ModuleList(
            [SiTUGLU(latent_dim, 2 * latent_dim, beta1, beta2) for _ in range(num_routed)]
        )
        self.u_norm = RMSNorm(latent_dim)
        self.w_up = nn.Linear(latent_dim, d_model, bias=False)

        self.router = nn.Linear(d_model, num_routed)
        self.register_buffer("route_bias", torch.zeros(num_routed))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        xf = x.reshape(B * T, C)

        scores = torch.sigmoid(self.router(xf))                       # [N, E]
        _, idx = torch.topk(scores + self.route_bias, self.top_k, dim=-1)
        mask = torch.zeros_like(scores).scatter_(-1, idx, 1.0)
        # Bias steers *selection* only; weights come from raw scores (Eq. 13)
        p = scores * mask
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-9)

        z = self.w_down(xf)                                           # [N, latent]
        u = torch.zeros_like(z)
        for e, expert in enumerate(self.experts):
            u = u + p[:, e].unsqueeze(-1) * expert(z)
        routed = self.w_up(self.u_norm(u))

        y = routed
        for expert in self.shared:
            y = y + expert(xf)

        if self.training:
            with torch.no_grad():
                load = mask.sum(dim=0)
                self.route_bias += self.gamma * torch.sign(load.mean() - load)

        return self.drop(y.view(B, T, C)), x.new_tensor(0.0)


class BlockAttnRes(nn.Module):
    """Block Attention Residuals: per-layer pseudo-query over depth (Eqs. 8-10).

    Sources are the token embedding (b0), completed block sums, and the running
    partial sum of the current block; keys are RMSNorm'd so loud layers can't
    dominate, values stay raw.
    """

    def __init__(self, d_model: int, n_slots: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.key_norm = RMSNorm(d_model)

    def forward(self, slot: int, sources: list[torch.Tensor]) -> torch.Tensor:
        stack = torch.stack(sources, dim=2)                    # [B, T, S, d]
        keys = self.key_norm(stack)
        logits = torch.einsum("btsd,d->bts", keys, self.queries[slot])
        weights = F.softmax(logits, dim=-1)
        return torch.einsum("bts,btsd->btd", weights, stack)


class _DenseFFN(nn.Module):
    """Full-width SiTU-GLU MLP for the single dense layer (paper Table 1)."""

    def __init__(self, d_model: int, d_ff: int, beta1: float, beta2: float, dropout: float):
        super().__init__()
        self.net = SiTUGLU(d_model, d_ff, beta1, beta2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.drop(self.net(x)), x.new_tensor(0.0)


class MTPModule(nn.Module):
    """Multi-Token Prediction: predict token t+2 from h_t and emb(x_{t+1}).

    One extra layer (paper Table 1: 1 MTP layer) sharing the embedding and LM
    head with the backbone; its CE rides the aux-loss channel, so the trainer
    contract is unchanged.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 beta1: float = 4.0, beta2: float = 25.0):
        super().__init__()
        self.norm_h = RMSNorm(d_model)
        self.norm_e = RMSNorm(d_model)
        self.proj = nn.Linear(2 * d_model, d_model)
        self.attn = GatedAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = SiTUGLU(d_model, 2 * d_model, beta1, beta2)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, next_emb: torch.Tensor) -> torch.Tensor:
        x = self.proj(torch.cat([self.norm_h(h), self.norm_e(next_emb)], dim=-1))
        x = self.norm1(x + self.attn(x))
        return self.norm2(x + self.ffn(x))


class _KimiSublayers(nn.Module):
    """One attention layer + its paired LatentMoE FFN, post-LN like the repo's blocks."""

    def __init__(self, attn: nn.Module, ffn: nn.Module, d_model: int, dropout: float):
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.norm1(x + self.drop(self.attn(x)))
        ffn_out, aux = self.ffn(x)
        x = self.norm2(x + self.drop(ffn_out))
        return x, aux


class KimiLM(nn.Module):
    """Mini Kimi K3 language model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 8,
        d_ff: int = 256,
        seq_len: int = 128,
        dropout: float = 0.1,
        kda_per_block: int = 3,
        kda_g_min: float = -5.0,
        kda_conv_kernel: int = 3,
        kda_decay_rank: int = 16,
        use_attn_res: bool = True,
        attn_res_block_size: int = 4,
        latent_dim: int = 64,
        num_routed: int = 8,
        router_top_k: int = 2,
        num_shared: int = 2,
        situ_beta1: float = 4.0,
        situ_beta2: float = 25.0,
        balance_gamma: float = 0.01,
        mtp_weight: float = 0.1,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)  # NoPE: no pos_embed
        self.seq_len = seq_len
        self.use_attn_res = use_attn_res
        self.block_size = attn_res_block_size
        self.mtp_weight = mtp_weight

        hybrid = kda_per_block + 1  # e.g. KDA,KDA,KDA,GatedAttention

        def make_attn(i: int) -> nn.Module:
            if i % hybrid < kda_per_block:
                return KDALayer(d_model, n_heads, kda_g_min, kda_conv_kernel, kda_decay_rank)
            return GatedAttention(d_model, n_heads, dropout)

        def make_ffn(i: int) -> nn.Module:
            if i == 0:  # paper Table 1: 1 dense layer before the MoE stack
                return _DenseFFN(d_model, d_ff, situ_beta1, situ_beta2, dropout)
            return StableLatentMoE(
                d_model, d_ff, latent_dim, num_routed, router_top_k,
                num_shared, situ_beta1, situ_beta2, balance_gamma, dropout,
            )

        attns = [make_attn(i) for i in range(n_layers)]
        attns.append(GatedAttention(d_model, n_heads, dropout))  # final layer is always global
        self.layers = nn.ModuleList(
            [_KimiSublayers(a, make_ffn(i), d_model, dropout) for i, a in enumerate(attns)]
        )
        self.mtp = MTPModule(d_model, n_heads, dropout, situ_beta1, situ_beta2)
        # one pseudo-query per layer + one for the final aggregation
        self.attn_res = BlockAttnRes(d_model, len(self.layers) + 1) if use_attn_res else None

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits [B, T, V], total_aux_loss scalar)."""
        emb = self.token_embed(tokens)
        total_aux = emb.new_tensor(0.0)

        if self.attn_res is None:
            x = emb
            for layer in self.layers:
                x, aux = layer(x)
                total_aux = total_aux + aux
        else:
            sources = [emb]        # b0 = embedding
            partial = torch.zeros_like(emb)
            flushed = True
            for i, layer in enumerate(self.layers):
                in_block = [partial] if i % self.block_size != 0 else []
                h = self.attn_res(i, sources + in_block)
                out, aux = layer(h)
                total_aux = total_aux + aux
                partial = partial + (out - h)  # this layer's residual contribution
                flushed = False
                if (i + 1) % self.block_size == 0:
                    sources.append(partial)
                    partial = torch.zeros_like(emb)
                    flushed = True
            if not flushed:  # trailing partial block
                sources.append(partial)
            x = self.attn_res(len(self.layers), sources)  # final aggregation over blocks

        x = self.norm(x)
        logits = self.lm_head(x)

        # MTP: predict x[t+2] from prefix<=t plus the teacher-forced next token
        T = tokens.shape[1]
        if self.training and T > 2:
            h2 = self.mtp(x[:, :-1], emb[:, 1:])
            mtp_logits = self.lm_head(h2)[:, :-1]           # positions predicting t+2
            mtp_ce = F.cross_entropy(
                mtp_logits.reshape(-1, mtp_logits.shape[-1]), tokens[:, 2:].reshape(-1)
            )
            total_aux = total_aux + self.mtp_weight * mtp_ce

        return logits, total_aux

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
