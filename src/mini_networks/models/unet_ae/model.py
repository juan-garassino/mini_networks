"""UNet autoencoder based on the segmentation UNet."""
from __future__ import annotations

from mini_networks.models.segmentation.unet import SegUNet


class UNetAutoencoder(SegUNet):
    def __init__(self, base_channels: int = 32):
        super().__init__(in_channels=1, out_channels=1, base_channels=base_channels)
