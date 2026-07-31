"""Config for the mini NeRF."""
from __future__ import annotations

from mini_networks.core.config import BaseConfig


class NerfConfig(BaseConfig):
    """Mini NeRF (arXiv 2003.08934) as a zoo entry.

    A voxelized MNIST digit as the 3D scene; ground-truth views are rendered
    EXACTLY (fixed 256 samples, never tier-capped) so all tiers train on the
    same target. Train/test views are interleaved azimuths — held-out novel
    views measure interpolation, the honest NeRF test. `timesteps` is reused
    as the MODEL's samples-per-ray budget (S 25 / M 200 via
    effective_timesteps, exactly like a diffusion chain length).
    """

    model_name: str = "nerf"

    image_size: int = 32       # rendered view resolution
    n_azimuths: int = 40       # orbit cameras in [0, 360); fast_demo drops to 8
    test_every: int = 4        # every 4th azimuth is held out (interleaved split)
    elevation_deg: float = 35.0  # fixed; grazing views would be background-dominated
    radius: float = 2.5        # camera orbit radius (scene lives in [-1, 1]^3)
    near: float = 1.0
    far: float = 4.0
    depth_layers: int = 8      # digit extrusion depth (voxels), colored by depth
    sigma_scale: float = 30.0  # voxel occupancy -> density

    pe_freqs: int = 6          # positional-encoding frequencies
    hidden_dim: int = 128
    n_layers: int = 5
    timesteps: int = 200       # samples per ray at L; tiers cap via effective_timesteps
    rays_per_batch: int = 1024  # ray minibatch (NOT batch_size — its M cap of 64 would starve training)
