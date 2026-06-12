"""UNet autoencoder based on the segmentation UNet.

An autoencoder learns to reproduce its input through a bottleneck, forcing
a compressed internal representation; trained on clean targets from noisy
or masked inputs it becomes a denoiser. This one reuses SegUNet unchanged —
the only difference from segmentation is the training target: the model
regresses the input image itself (out_channels=1, sigmoid output) instead
of a label mask.

Inherited architecture (base_channels=32):

    enc1(1->32) 28x28 --skip------------------+
      pool -> enc2(32->64) 14x14 --skip--+    |
        pool -> bottleneck(64->128) 7x7  |    |
        upconv -> cat -> dec2(128->64) --+    |
      upconv -> cat -> dec1(64->32) ----------+
    Conv1x1(32->1) -> sigmoid

The educational catch: a UNet is almost too good an autoencoder. The skip
connections let fine detail bypass the 7x7 bottleneck entirely, so the
network can reconstruct nearly perfectly without learning a meaningful
compressed code — useful for denoising, useless for representation
learning. Compare with the VAE, which has no skips and a stochastic
bottleneck. Deliberately simplified: plain identity reconstruction with no
noise/masking corruption, and the bottleneck code is never exposed as an
embedding.
"""
from __future__ import annotations

from mini_networks.models.segmentation.unet import SegUNet


class UNetAutoencoder(SegUNet):
    def __init__(self, base_channels: int = 32):
        super().__init__(in_channels=1, out_channels=1, base_channels=base_channels)
