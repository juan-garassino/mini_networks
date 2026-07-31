"""Synthetic voxel scene + camera rays + exact ground-truth renderer.

The scene is one seeded MNIST digit extruded into a 28x28xD voxel slab and
colored by depth layer (so RGB carries real information). Ground truth is
rendered with a FIXED high sample count, independent of training tiers —
S/M/L must all chase the same target, and the model can actually reach it.
Everything is vectorized torch: no per-ray Python loops (CI renders this on
CPU at S tier).
"""
from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, Dataset

GT_SAMPLES = 256  # exact-GT sample count — NEVER tier-capped


def build_voxels(data_root: str, seed: int, depth_layers: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (occupancy [28,28,D], rgb [28,28,D,3]) for one seeded MNIST digit."""
    import torchvision
    import torchvision.transforms as T

    ds = torchvision.datasets.MNIST(root=data_root, train=True, download=True,
                                    transform=T.ToTensor())
    g = torch.Generator().manual_seed(seed)
    img, _ = ds[int(torch.randint(0, len(ds), (1,), generator=g))]
    ink = img.squeeze(0)                                   # [28, 28] in [0, 1]
    occ = (ink > 0.3).float().unsqueeze(-1).expand(-1, -1, depth_layers).contiguous()
    # depth colormap: front layers blue -> back layers yellow
    z = torch.linspace(0, 1, depth_layers)
    rgb = torch.stack([z, z, 1 - z], dim=-1)               # [D, 3]
    rgb = rgb.view(1, 1, depth_layers, 3).expand(28, 28, -1, -1).contiguous()
    return occ, rgb


def make_rays(azimuth_deg: float, elevation_deg: float, size: int,
              radius: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Pinhole camera on an orbit, looking at the origin.
    Returns (rays_o [size*size, 3], rays_d [size*size, 3])."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    eye = torch.tensor([
        radius * math.cos(el) * math.cos(az),
        radius * math.cos(el) * math.sin(az),
        radius * math.sin(el),
    ])
    forward = -eye / eye.norm()
    up = torch.tensor([0.0, 0.0, 1.0])
    right = torch.linalg.cross(forward, up)
    right = right / right.norm()
    cam_up = torch.linalg.cross(right, forward)
    fov = 0.7  # ~40 degrees
    xs = torch.linspace(-1, 1, size) * math.tan(fov / 2)
    ys = torch.linspace(1, -1, size) * math.tan(fov / 2)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    dirs = gx.unsqueeze(-1) * right + gy.unsqueeze(-1) * cam_up + forward
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    rays_d = dirs.reshape(-1, 3)
    rays_o = eye.expand_as(rays_d)
    return rays_o, rays_d


def _voxel_lookup(pts: torch.Tensor, occ: torch.Tensor, rgb: torch.Tensor,
                  depth_layers: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-neighbor voxel query. pts [N, 3] in world space; the slab spans
    x,y in [-1,1], z in [-z_half, z_half] with z_half = depth_layers/28."""
    z_half = depth_layers / 28.0
    ix = ((pts[:, 0] + 1) / 2 * 28).long().clamp(0, 27)
    iy = ((1 - (pts[:, 1] + 1) / 2) * 28).long().clamp(0, 27)
    iz = ((pts[:, 2] + z_half) / (2 * z_half) * depth_layers).long().clamp(0, depth_layers - 1)
    inside = (pts.abs()[:, :2] < 1).all(dim=1) & (pts[:, 2].abs() < z_half)
    sigma = occ[iy, ix, iz] * inside.float()
    color = rgb[iy, ix, iz] * inside.float().unsqueeze(-1)
    return sigma, color


def composite(sigma: torch.Tensor, color: torch.Tensor,
              t_vals: torch.Tensor) -> torch.Tensor:
    """Volume rendering: sigma/color [R, S(,3)], t_vals [R, S] -> rgb [R, 3]."""
    dt = torch.cat([t_vals[:, 1:] - t_vals[:, :-1],
                    torch.full_like(t_vals[:, :1], 1e-2)], dim=1)
    alpha = 1 - torch.exp(-sigma * dt)
    trans = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-10], dim=1), dim=1
    )[:, :-1]
    weights = alpha * trans
    return (weights.unsqueeze(-1) * color).sum(dim=1)


def render_gt(occ: torch.Tensor, rgb: torch.Tensor, azimuth_deg: float,
              config) -> torch.Tensor:
    """Exact GT view [size*size, 3] with the FIXED sample count."""
    rays_o, rays_d = make_rays(azimuth_deg, config.elevation_deg,
                               config.image_size, config.radius)
    t = torch.linspace(config.near, config.far, GT_SAMPLES).expand(rays_o.shape[0], -1)
    pts = rays_o.unsqueeze(1) + t.unsqueeze(-1) * rays_d.unsqueeze(1)   # [R, S, 3]
    sigma, color = _voxel_lookup(pts.reshape(-1, 3), occ, rgb, config.depth_layers)
    sigma = sigma.view(t.shape) * config.sigma_scale
    color = color.view(*t.shape, 3)
    return composite(sigma, color, t)


class NerfViewDataset(Dataset):
    """One item per camera view: (rays_o, rays_d, colors), all [size^2, 3].

    Interleaved azimuth split: every `test_every`-th view is held out — novel
    views between training views (interpolation, the honest NeRF test). The
    0/360 duplicate never exists because azimuths span [0, 360)."""

    def __init__(self, config, split: str = "train"):
        n_az = 8 if config.effective_fast_demo else config.n_azimuths
        azimuths = [360.0 * i / n_az for i in range(n_az)]
        pick_test = lambda i: i % config.test_every == config.test_every - 1
        wanted = [a for i, a in enumerate(azimuths)
                  if pick_test(i) == (split != "train")]
        occ, rgb = build_voxels(config.data_root, config.seed, config.depth_layers)
        self.azimuths = wanted
        self.views = []
        for az in wanted:
            rays_o, rays_d = make_rays(az, config.elevation_deg,
                                       config.image_size, config.radius)
            colors = render_gt(occ, rgb, az, config)
            self.views.append((rays_o, rays_d, colors))

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, idx: int):
        return self.views[idx]


def make_nerf_dataloader(config, split: str = "train") -> DataLoader:
    # one item per view; the trainer flattens views into a ray pool itself
    return DataLoader(NerfViewDataset(config, split), batch_size=1, shuffle=False)
