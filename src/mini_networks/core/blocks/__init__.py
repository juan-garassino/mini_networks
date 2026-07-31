"""Shared building blocks — the few pieces of code models genuinely share.

The zoo's convention is SELF-CONTAINED model files (copy-paste over
abstraction, so each model.py reads as one lesson); a block graduates here
only when it is an exact duplicate or an awkward cross-model import.

Inventory (consumers per block):
  cnn.py        ConvBNReLU            classifier, vision_embed, mobilenet, convnext
                DepthwiseSeparable    mobilenet
  mlp.py        MLP                   tabular_classifier
  norm.py       RMSNorm               kimi, deepseek
  ema.py        EMA                   diffusion, gan
  attention.py  TransformerEncoderBlock   (no consumers — deletion candidate,
                                          kept pending an owner decision)

Same-mechanism, deliberately NOT unified (variants differ): dice_loss
(segmentation vs sam reductions), InfoNCE (simclr vs clip), Fourier
features (sam random-gaussian vs nerf geometric), SiTU-GLU (kimi-only).
The conceptual sharing is recorded in core/taxonomy.py instead.
"""
