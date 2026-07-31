"""Config for the masked-diffusion text LM (LLaDA-mini)."""
from __future__ import annotations

from mini_networks.models.transformer.config import TransformerConfig


class TextDiffusionConfig(TransformerConfig):
    """Masked-diffusion char-LM (LLaDA, arXiv 2502.09992) as a zoo entry.

    Same corpus/tokenizer fields as `transformer` (reuses the text dataloader),
    but the model is BIDIRECTIONAL and generates by iterative parallel
    unmasking instead of left-to-right sampling — "same architecture, two
    generation orders". eval_loss is masked CE at t=0.5, its own band (NOT
    comparable to autoregressive CE).
    """

    model_name: str = "text_diffusion"

    mask_ratio_min: float = 0.05  # t ~ U(min, 1); avoids the degenerate 1/t blow-up at t->0
    timesteps: int = 64           # unmask rounds at L; effective_timesteps caps per tier (S 25)
