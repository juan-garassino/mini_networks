"""Kimi trainer — TransformerTrainer with a KimiLM build."""
from __future__ import annotations

from mini_networks.models.kimi.config import KimiConfig
from mini_networks.models.kimi.model import KimiLM
from mini_networks.models.transformer.trainer import (
    TransformerTrainer,
    make_transformer_dataloader,
)

make_kimi_dataloader = make_transformer_dataloader


class KimiTrainer(TransformerTrainer):
    """Inherits train/evaluate/infer/load_checkpoint — only the model differs."""

    def _build(self, config: KimiConfig) -> KimiLM:  # type: ignore[override]
        return KimiLM(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            seq_len=config.seq_len,
            dropout=config.dropout,
            kda_per_block=config.kda_per_block,
            kda_g_min=config.kda_g_min,
            kda_conv_kernel=config.kda_conv_kernel,
            kda_decay_rank=config.kda_decay_rank,
            use_attn_res=config.use_attn_res,
            attn_res_block_size=config.attn_res_block_size,
            latent_dim=config.latent_dim,
            num_routed=config.num_routed,
            router_top_k=config.router_top_k,
            num_shared=config.num_shared,
            situ_beta1=config.situ_beta1,
            situ_beta2=config.situ_beta2,
            balance_gamma=config.balance_gamma,
            mtp_weight=config.mtp_weight,
        ).to(config.device)
