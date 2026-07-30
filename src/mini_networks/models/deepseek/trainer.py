"""Deepseek trainer — TransformerTrainer with a DeepseekLM build."""
from __future__ import annotations

from mini_networks.models.deepseek.config import DeepseekConfig
from mini_networks.models.deepseek.model import DeepseekLM
from mini_networks.models.transformer.trainer import (
    TransformerTrainer,
    make_transformer_dataloader,
)

make_deepseek_dataloader = make_transformer_dataloader


class DeepseekTrainer(TransformerTrainer):
    """Inherits train/evaluate/infer/load_checkpoint — only the model differs."""

    def _build(self, config: DeepseekConfig) -> DeepseekLM:  # type: ignore[override]
        return DeepseekLM(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            seq_len=config.seq_len,
            dropout=config.dropout,
            csa_m=config.csa_m,
            csa_top_k=config.csa_top_k,
            hca_m=config.hca_m,
            hca_every=config.hca_every,
            kv_dim=config.kv_dim,
            n_win=config.n_win,
            rope_dims=config.rope_dims,
            use_mhc=config.use_mhc,
            n_hc=config.n_hc,
            sinkhorn_iters=config.sinkhorn_iters,
            ds_num_experts=config.ds_num_experts,
            ds_top_k=config.ds_top_k,
            ds_num_shared=config.ds_num_shared,
            balance_gamma=config.balance_gamma,
            mtp_weight=config.mtp_weight,
        ).to(config.device)
