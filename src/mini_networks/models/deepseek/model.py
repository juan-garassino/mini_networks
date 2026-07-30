"""Mini DeepSeek V4: compressed hybrid attention LM with mHC and DeepSeekMoE.

Key idea (DeepSeek V4 technical report, arXiv 2606.19348): break the
long-context efficiency barrier by compressing the KV cache along the
sequence dimension — Compressed Sparse Attention (CSA) pools every m tokens
into one KV entry and then attends sparsely to the top-k entries; Heavily
Compressed Attention (HCA) pools every m' >> m tokens and attends densely.
Both add a sliding-window branch for local detail and a learnable attention
sink. Residual connections are upgraded to Manifold-Constrained
Hyper-Connections (mHC): the residual stream is widened to n_hc x d and mixed
per sublayer by a doubly-stochastic matrix (Sinkhorn projection keeps the
spectral norm <= 1, so deep stacks stay stable). FFNs are DeepSeekMoE
(fine-grained routed + shared experts, sqrt-softplus affinity, aux-loss-free
bias balancing + a slight sequence-wise balance loss), with the FIRST block's
FFN hash-routed by token id (Roller et al.). A Multi-Token Prediction module
predicts token t+2 as an auxiliary objective.

This implementation (defaults): 8 sublayer pairs — attention alternating
3 CSA : 1 HCA — each pair's attention and FFN wrapped separately by mHC
(Eq. 1: X_{l+1} = B_l X_l + C_l F_l(A_l X_l)). No absolute position
embeddings: partial RoPE on the last rope_dims of queries and KV entries,
and inverse RoPE (position -t) on the attention outputs so they carry
relative position only (paper section 2.3.3).

Key equations:
  Compressor    C_i = sum_j softmax_j(Z + B_pos) . C_j over each m-token block
  CSA indexer   I_ts = sum_h w_h ReLU(q_h . comp_s), Top-k entries kept
  Sink          softmax over [entries, window, z'_h] — heads may attend "nowhere"
  mHC           A = sigma(.), C = 2 sigma(.), B = Sinkhorn(exp(.)) doubly stochastic
  MoE affinity  s = sqrt(softplus(router(x))), bias enters Top-k selection only

Deliberately simplified vs the paper: single output projection (grouped
output projection is a width artifact at c*n_h ~ 16k dims); the indexer
shares the token compressor with core attention (V4 trains a separate
lightning-indexer compressor, FP4); one hash-routed block instead of
"the initial several"; no Muon, no FP8/FP4, no million-token machinery.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm (torch<2.4 has no nn.RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def _rope(x: torch.Tensor, pos: torch.Tensor, n_dims: int) -> torch.Tensor:
    """Rotate the last n_dims of x by position; pos indexes x's dim 1 (may be negative)."""
    half = n_dims // 2
    inv = 10000.0 ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = pos.to(torch.float32)[:, None] * inv                     # [S, half]
    shape = [1, x.shape[1]] + [1] * (x.dim() - 3) + [half]
    cos, sin = ang.cos().view(shape), ang.sin().view(shape)
    keep, rot = x[..., :-n_dims], x[..., -n_dims:]
    r1, r2 = rot[..., :half], rot[..., half:]
    return torch.cat([keep, r1 * cos - r2 * sin, r1 * sin + r2 * cos], dim=-1)


def sinkhorn(m: torch.Tensor, iters: int) -> torch.Tensor:
    """Project a positive matrix onto the doubly-stochastic manifold (rows/cols sum 1)."""
    for _ in range(iters):
        m = m / (m.sum(dim=-1, keepdim=True) + 1e-9)
        m = m / (m.sum(dim=-2, keepdim=True) + 1e-9)
    return m


class TokenCompressor(nn.Module):
    """Pool every m tokens into one c-dim KV entry via learned weights + position biases."""

    def __init__(self, d_model: int, c: int, m: int):
        super().__init__()
        self.m = m
        self.w_kv = nn.Linear(d_model, c, bias=False)
        self.w_z = nn.Linear(d_model, c, bias=False)
        self.pos_bias = nn.Parameter(torch.zeros(m, c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        nb = T // self.m
        if nb == 0:
            return x.new_zeros(B, 0, self.w_kv.out_features)
        x = x[:, : nb * self.m]
        c = self.w_kv(x).view(B, nb, self.m, -1)
        z = self.w_z(x).view(B, nb, self.m, -1) + self.pos_bias
        return (F.softmax(z, dim=2) * c).sum(dim=2)


class CompressedAttention(nn.Module):
    """CSA (top_k set) or HCA (top_k=None): shared-KV MQA over compressed entries
    + sliding-window branch + attention sink."""

    def __init__(
        self,
        d_model: int,
        c: int,
        n_heads: int,
        m: int,
        top_k: int | None,
        n_win: int,
        rope_dims: int,
        n_idx_heads: int = 2,
    ):
        super().__init__()
        self.c = c
        self.m = m
        self.n_heads = n_heads
        self.top_k = top_k
        self.n_win = n_win
        self.rope_dims = rope_dims

        self.compressor = TokenCompressor(d_model, c, m)
        self.w_dq = nn.Linear(d_model, c, bias=False)     # latent query down-proj
        self.w_uq = nn.Linear(c, n_heads * c, bias=False)  # up-proj to per-head queries
        self.w_win = nn.Linear(d_model, c, bias=False)     # sliding-window KV entries
        self.q_norm = RMSNorm(c)
        self.kv_norm = RMSNorm(c)
        self.sink = nn.Parameter(torch.zeros(n_heads))     # z'_h (Eq. 27)
        if top_k is not None:  # lightning indexer (Eqs. 13-17)
            self.w_iq = nn.Linear(c, n_idx_heads * c, bias=False)
            self.w_iw = nn.Linear(d_model, n_idx_heads, bias=False)
        self.w_o = nn.Linear(n_heads * c, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, c = self.n_heads, self.c
        t_pos = torch.arange(T, device=x.device)

        comp = self.kv_norm(self.compressor(x))            # [B, nb, c]
        nb = comp.shape[1]
        # entry position = its block's last token (relative distances stay sane)
        comp = _rope(comp, (torch.arange(nb, device=x.device) + 1) * self.m - 1, self.rope_dims)

        c_q = self.w_dq(x)
        q = self.q_norm(self.w_uq(c_q).view(B, T, H, c))
        q = _rope(q, t_pos, self.rope_dims)

        win = _rope(self.kv_norm(self.w_win(x)), t_pos, self.rope_dims)  # [B, T, c]

        # --- compressed-entry logits, causal over strictly-preceding blocks ---
        logits_c = torch.einsum("bthc,bsc->bhts", q, comp) / c**0.5
        block_of_t = t_pos // self.m                        # query t's own block
        causal_c = torch.arange(nb, device=x.device)[None, :] >= block_of_t[:, None]  # [T, nb]
        logits_c = logits_c.masked_fill(causal_c[None, None], float("-inf"))

        if self.top_k is not None and nb > 0:
            qi = self.w_iq(c_q).view(B, T, -1, c)
            wi = self.w_iw(x)                               # [B, T, nI]
            idx_scores = torch.einsum(
                "btn,btns->bts", wi, F.relu(torch.einsum("btnc,bsc->btns", qi, comp))
            )
            idx_scores = idx_scores.masked_fill(causal_c[None], float("-inf"))
            k = min(self.top_k, nb)
            sel = torch.topk(idx_scores, k, dim=-1).indices
            keep = torch.zeros_like(idx_scores, dtype=torch.bool).scatter_(-1, sel, True)
            logits_c = logits_c.masked_fill(~keep[:, None], float("-inf"))

        # --- sliding-window logits over raw tokens (covers the current block) ---
        logits_w = torch.einsum("bthc,bjc->bhtj", q, win) / c**0.5
        j_pos = t_pos[None, :]
        band = (j_pos <= t_pos[:, None]) & (j_pos > t_pos[:, None] - self.n_win)
        logits_w = logits_w.masked_fill(~band[None, None], float("-inf"))

        sink = self.sink.view(1, H, 1, 1).expand(B, H, T, 1)
        weights = F.softmax(torch.cat([logits_c, logits_w, sink], dim=-1), dim=-1)
        w_c, w_w = weights[..., :nb], weights[..., nb : nb + T]  # sink weight is dropped

        o = torch.einsum("bhts,bsc->bthc", w_c, comp) + torch.einsum("bhtj,bjc->bthc", w_w, win)
        o = _rope(o, -t_pos, self.rope_dims)  # cancel absolute positions carried by entries
        return self.w_o(o.reshape(B, T, H * c))


class _SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w_g = nn.Linear(dim, hidden)
        self.w_u = nn.Linear(dim, hidden)
        self.w_d = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_d(F.silu(self.w_g(x)) * self.w_u(x))


class DeepSeekMoEFFN(nn.Module):
    """Fine-grained routed + shared experts; sqrt(softplus) affinity; aux-loss-free
    bias balancing + slight sequence-wise balance loss. hash_route=True assigns
    experts by token id instead (Roller et al.), used in the first block."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        num_shared: int = 1,
        balance_gamma: float = 0.01,
        seq_balance_alpha: float = 1e-3,
        hash_route: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gamma = balance_gamma
        self.alpha = seq_balance_alpha
        self.hash_route = hash_route

        self.shared = nn.ModuleList([_SwiGLU(d_model, d_ff) for _ in range(num_shared)])
        # fine-grained: routed experts are half-width
        self.experts = nn.ModuleList([_SwiGLU(d_model, d_ff // 2) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts)
        self.register_buffer("route_bias", torch.zeros(num_experts))

    def forward(
        self, x: torch.Tensor, token_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        xf = x.reshape(B * T, C)
        aux = x.new_tensor(0.0)

        if self.hash_route:
            assert token_ids is not None
            idx = (token_ids.reshape(-1) % self.num_experts).unsqueeze(-1)
            p = torch.zeros(B * T, self.num_experts, device=x.device).scatter_(-1, idx, 1.0)
        else:
            affinity = torch.sqrt(F.softplus(self.router(xf)) + 1e-9)   # [N, E]
            _, sel = torch.topk(affinity + self.route_bias, self.top_k, dim=-1)
            mask = torch.zeros_like(affinity).scatter_(-1, sel, 1.0)
            p = affinity * mask
            p = p / (p.sum(dim=-1, keepdim=True) + 1e-9)

            if self.training:
                with torch.no_grad():
                    load = mask.sum(dim=0)
                    self.route_bias += self.gamma * torch.sign(load.mean() - load)
                # sequence-wise balance: E[f_e * P_e] per sequence (V3-style)
                E = self.num_experts
                f = mask.view(B, T, E).mean(dim=1) * E / self.top_k
                s_norm = affinity / (affinity.sum(dim=-1, keepdim=True) + 1e-9)
                P = s_norm.view(B, T, E).mean(dim=1)
                aux = aux + self.alpha * (f * P).sum(dim=-1).mean()

        routed = torch.zeros_like(xf)
        for e, expert in enumerate(self.experts):
            routed = routed + p[:, e].unsqueeze(-1) * expert(xf)
        y = routed
        for expert in self.shared:
            y = y + expert(xf)
        return y.view(B, T, C), aux


class MHCMixer(nn.Module):
    """Generates A (input), B (residual, doubly stochastic), C (output) per token (Eqs. 3-8)."""

    def __init__(self, d_model: int, n: int, sinkhorn_iters: int = 10):
        super().__init__()
        self.n = n
        self.iters = sinkhorn_iters
        self.norm = RMSNorm(n * d_model)
        self.w_pre = nn.Linear(n * d_model, n, bias=False)
        self.w_res = nn.Linear(n * d_model, n * n, bias=False)
        self.w_post = nn.Linear(n * d_model, n, bias=False)
        self.a_pre = nn.Parameter(torch.tensor(0.01))
        self.a_res = nn.Parameter(torch.tensor(0.01))
        self.a_post = nn.Parameter(torch.tensor(0.01))
        self.s_pre = nn.Parameter(torch.zeros(n))
        self.s_res = nn.Parameter(2.0 * torch.eye(n))  # B starts identity-leaning
        self.s_post = nn.Parameter(torch.zeros(n))

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B_, T, n, d = X.shape
        xh = self.norm(X.reshape(B_, T, n * d))
        A = torch.sigmoid(self.a_pre * self.w_pre(xh) + self.s_pre)          # [B, T, n]
        raw = (self.a_res * self.w_res(xh)).view(B_, T, n, n) + self.s_res
        Bm = sinkhorn(torch.exp(raw.clamp(max=10.0)), self.iters)            # [B, T, n, n]
        C = 2.0 * torch.sigmoid(self.a_post * self.w_post(xh) + self.s_post)  # [B, T, n]
        return A, Bm, C


class _Branch(nn.Module):
    """Pre-norm sublayer branch F_l: mHC supplies the skip (Eq. 1), so F returns
    only the branch output."""

    def __init__(self, inner: nn.Module, d_model: int, dropout: float, is_ffn: bool):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.inner = inner
        self.drop = nn.Dropout(dropout)
        self.is_ffn = is_ffn

    def forward(
        self, h: torch.Tensor, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_ffn:
            out, aux = self.inner(self.norm(h), token_ids)
            return self.drop(out), aux
        return self.drop(self.inner(self.norm(h))), h.new_tensor(0.0)


class MTPModule(nn.Module):
    """Multi-Token Prediction: predict token t+2 from h_t and emb(x_{t+1}) (V3 design).

    Shares the embedding and LM head with the backbone; contributes a weighted
    CE term through the aux-loss channel, so the trainer contract is unchanged.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm_h = RMSNorm(d_model)
        self.norm_e = RMSNorm(d_model)
        self.proj = nn.Linear(2 * d_model, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = _SwiGLU(d_model, 2 * d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, next_emb: torch.Tensor) -> torch.Tensor:
        x = self.proj(torch.cat([self.norm_h(h), self.norm_e(next_emb)], dim=-1))
        T = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        a, _ = self.attn(x, x, x, attn_mask=mask, is_causal=True)
        x = self.norm1(x + a)
        return self.norm2(x + self.ffn(x))


class DeepseekLM(nn.Module):
    """Mini DeepSeek V4 language model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 8,
        d_ff: int = 256,
        seq_len: int = 128,
        dropout: float = 0.1,
        csa_m: int = 4,
        csa_top_k: int = 8,
        hca_m: int = 16,
        hca_every: int = 4,
        kv_dim: int = 64,
        n_win: int = 16,
        rope_dims: int = 16,
        use_mhc: bool = True,
        n_hc: int = 2,
        sinkhorn_iters: int = 10,
        ds_num_experts: int = 8,
        ds_top_k: int = 2,
        ds_num_shared: int = 1,
        balance_gamma: float = 0.01,
        mtp_weight: float = 0.1,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)  # no absolute positions: partial RoPE
        self.seq_len = seq_len
        self.use_mhc = use_mhc
        self.n_hc = n_hc
        self.mtp_weight = mtp_weight

        branches: list[_Branch] = []
        for i in range(n_layers):
            is_hca = (i + 1) % hca_every == 0
            attn = CompressedAttention(
                d_model, kv_dim, n_heads,
                m=hca_m if is_hca else csa_m,
                top_k=None if is_hca else csa_top_k,
                n_win=n_win, rope_dims=rope_dims,
            )
            ffn = DeepSeekMoEFFN(
                d_model, d_ff, ds_num_experts, ds_top_k, ds_num_shared,
                balance_gamma, hash_route=(i == 0),  # first block: hash routing by token id
            )
            branches.append(_Branch(attn, d_model, dropout, is_ffn=False))
            branches.append(_Branch(ffn, d_model, dropout, is_ffn=True))
        self.branches = nn.ModuleList(branches)
        if use_mhc:
            self.mixers = nn.ModuleList(
                [MHCMixer(d_model, n_hc, sinkhorn_iters) for _ in branches]
            )

        self.mtp = MTPModule(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits [B, T, V], total_aux_loss scalar)."""
        B, T = tokens.shape
        emb = self.token_embed(tokens)
        total_aux = emb.new_tensor(0.0)

        if self.use_mhc:
            X = emb.unsqueeze(2).expand(B, T, self.n_hc, emb.shape[-1]).contiguous()
            for branch, mixer in zip(self.branches, self.mixers):
                A, Bm, C = mixer(X)
                h = torch.einsum("btn,btnd->btd", A, X)
                out, aux = branch(h, tokens)
                total_aux = total_aux + aux
                X = torch.einsum("btij,btjd->btid", Bm, X) + C.unsqueeze(-1) * out.unsqueeze(2)
            x = X.mean(dim=2)
        else:
            x = emb
            for branch in self.branches:
                out, aux = branch(x, tokens)
                total_aux = total_aux + aux
                x = x + out

        x = self.norm(x)
        logits = self.lm_head(x)

        # MTP: predict x[t+2] from prefix<=t plus the teacher-forced next token
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
