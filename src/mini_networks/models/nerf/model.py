"""Mini NeRF: a scene as a continuous function, supervised through rendering.

Key idea (Mildenhall et al., arXiv 2003.08934): represent a 3D scene not as
voxels or meshes but as a FUNCTION — an MLP mapping a 3D point to (density,
color) — and train it with no 3D supervision at all: march rays from posed
cameras through the field, alpha-composite the samples into pixels (volume
rendering is differentiable), and minimize photometric error against the
training views. Novel views then come for free by rendering from unseen
poses. The enabling trick is POSITIONAL ENCODING: raw (x,y,z) inputs bias
MLPs toward low-frequency functions; mapping each coordinate through
sin/cos at geometrically-spaced frequencies lets the network represent
sharp spatial detail.

This implementation (defaults): a 5-layer MLP(128) with L=6 PE frequencies
on a voxelized MNIST digit (28x28x8 slab, depth-colored), 30 training
azimuths / 10 interleaved held-out azimuths at fixed 35-degree elevation.
Samples-per-ray reuses the tier `timesteps` budget (S 25 / M 200) —
stratified during training, bin midpoints at eval so PSNR is deterministic.

Key equations:
  PE(p)   = [sin(2^k pi p), cos(2^k pi p)]  for k = 0..L-1, per coordinate
  render  C = sum_i T_i (1 - exp(-sigma_i dt_i)) c_i,
          T_i = exp(-sum_{j<i} sigma_j dt_j)
  loss    MSE(C_pred, C_gt);  PSNR = -10 log10(MSE)

Deliberately simplified vs the paper: no view-direction input (the voxel
scene is Lambertian — nothing view-dependent to learn), no hierarchical
coarse/fine sampling, one scene per run. A learned voxel grid would beat
this MLP on this scene; the continuous-field representation is the lesson,
not the benchmark win.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MiniNeRF(nn.Module):
    def __init__(self, pe_freqs: int = 6, hidden_dim: int = 128, n_layers: int = 5):
        super().__init__()
        self.pe_freqs = pe_freqs
        in_dim = 3 * 2 * pe_freqs
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 4)  # sigma + rgb

    def encode(self, pts: torch.Tensor) -> torch.Tensor:
        freqs = 2.0 ** torch.arange(self.pe_freqs, device=pts.device) * torch.pi
        ang = pts.unsqueeze(-1) * freqs                     # [..., 3, L]
        return torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)

    def forward(self, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """pts [..., 3] -> (sigma [...], rgb [..., 3])."""
        out = self.head(self.trunk(self.encode(pts)))
        return torch.nn.functional.softplus(out[..., 0]), torch.sigmoid(out[..., 1:])


def render_rays(
    model: MiniNeRF,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    n_samples: int,
    near: float,
    far: float,
    stratified: bool = False,
    chunk: int = 16384,
) -> torch.Tensor:
    """Volume-render rays through the field. Returns rgb [R, 3].
    stratified=True jitters sample positions (training); False uses bin
    midpoints (deterministic eval)."""
    from mini_networks.models.nerf.scene import composite

    R = rays_o.shape[0]
    edges = torch.linspace(near, far, n_samples + 1, device=rays_o.device)
    lo, hi = edges[:-1], edges[1:]
    if stratified:
        u = torch.rand(R, n_samples, device=rays_o.device)
    else:
        u = torch.full((R, n_samples), 0.5, device=rays_o.device)
    t_vals = lo + (hi - lo) * u                             # [R, S]

    out = []
    for start in range(0, R, chunk):
        t = t_vals[start:start + chunk]
        o = rays_o[start:start + chunk]
        d = rays_d[start:start + chunk]
        pts = o.unsqueeze(1) + t.unsqueeze(-1) * d.unsqueeze(1)
        sigma, rgb = model(pts)
        out.append(composite(sigma, rgb, t))
    return torch.cat(out, dim=0)
