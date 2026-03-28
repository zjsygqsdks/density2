"""
renderer.py — Differentiable volume rendering for DustNeRF.

Implements:
  * stratified sampling along a ray
  * hierarchical importance sampling (coarse → fine)
  * volume rendering integral (Mildenhall et al., NeRF 2020)
  * dust density accumulation
  * temporal frame-index forwarding to the model
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from .model import DustNeRF


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def sample_stratified(
    rays_o: torch.Tensor,   # (B, 3)
    rays_d: torch.Tensor,   # (B, 3)
    near: float,
    far: float,
    n_samples: int,
    perturb: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stratified sampling of t-values along each ray.

    Returns
    -------
    pts : (B, n_samples, 3)  3-D sample points
    z   : (B, n_samples)     t-values along each ray
    """
    B = rays_o.shape[0]
    device = rays_o.device

    t = torch.linspace(0.0, 1.0, n_samples, device=device)
    z = near * (1.0 - t) + far * t                             # (n_samples,)
    z = z.unsqueeze(0).expand(B, -1)                           # (B, n_samples)

    if perturb:
        mids  = 0.5 * (z[:, 1:] + z[:, :-1])
        upper = torch.cat([mids, z[:, -1:]], dim=-1)
        lower = torch.cat([z[:, :1], mids],  dim=-1)
        noise = torch.rand_like(z)
        z = lower + (upper - lower) * noise

    pts = rays_o[:, None, :] + rays_d[:, None, :] * z[..., None]
    return pts, z


def sample_pdf(
    bins: torch.Tensor,     # (B, n_coarse-1) midpoints
    weights: torch.Tensor,  # (B, n_coarse-2) importance weights
    n_samples: int,
    det: bool = False,
) -> torch.Tensor:
    """
    Hierarchical importance sampling from a piecewise-constant PDF.

    Returns
    -------
    z_fine : (B, n_samples) new t-values sampled from the PDF
    """
    device = weights.device
    B = weights.shape[0]

    weights = weights + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)

    if det:
        u = torch.linspace(0.0, 1.0, n_samples, device=device)
        u = u.unsqueeze(0).expand(B, -1)
    else:
        u = torch.rand(B, n_samples, device=device)

    u = u.contiguous()
    inds  = torch.searchsorted(cdf.contiguous(), u, right=True)
    below = (inds - 1).clamp(min=0)
    above = inds.clamp(max=cdf.shape[-1] - 1)

    inds_g = torch.stack([below, above], dim=-1)

    cdf_g  = torch.gather(cdf.unsqueeze(1).expand(-1, n_samples, -1),  2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(-1, n_samples, -1), 2, inds_g)

    denom  = cdf_g[..., 1] - cdf_g[..., 0]
    denom  = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t      = (u - cdf_g[..., 0]) / denom
    z_fine = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return z_fine


# ---------------------------------------------------------------------------
# Volume rendering integral
# ---------------------------------------------------------------------------

def volume_render(
    sigma: torch.Tensor,      # (B, S)
    rgb: torch.Tensor,        # (B, S, 3)
    dust_prob: torch.Tensor,  # (B, S)
    z: torch.Tensor,          # (B, S)
    rays_d: torch.Tensor,     # (B, 3)
    white_bkgd: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute rendered colour, depth, opacity, and dust density.

    Returns
    -------
    rgb_map   : (B, 3)  rendered colour
    depth_map : (B,)    expected depth
    acc_map   : (B,)    accumulated opacity
    weights   : (B, S)  per-sample weights (for hierarchical sampling)
    dust_map  : (B,)    integrated dust density along ray
    """
    dists = z[:, 1:] - z[:, :-1]
    dists = torch.cat([dists, torch.full_like(dists[:, :1], 1e10)], dim=-1)
    dists = dists * torch.norm(rays_d[:, None, :], dim=-1)

    alpha = 1.0 - torch.exp(-sigma * dists)
    T = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]),
                   1.0 - alpha[:, :-1] + 1e-10], dim=-1),
        dim=-1,
    )
    weights = T * alpha

    rgb_map   = (weights[..., None] * rgb).sum(dim=1)
    depth_map = (weights * z).sum(dim=1)
    acc_map   = weights.sum(dim=1)
    dust_map  = (weights * dust_prob).sum(dim=1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[:, None])

    return rgb_map, depth_map, acc_map, weights, dust_map


# ---------------------------------------------------------------------------
# Full render pass (coarse + fine)
# ---------------------------------------------------------------------------

def render_rays(
    model_coarse: DustNeRF,
    model_fine: DustNeRF,
    rays_o: torch.Tensor,                        # (B, 3)
    rays_d: torch.Tensor,                        # (B, 3)
    near: float,
    far: float,
    n_coarse: int,
    n_fine: int,
    perturb: bool = True,
    white_bkgd: bool = False,
    frame_idx: Optional[torch.Tensor] = None,    # (B,) int64 frame indices
) -> dict:
    """
    Full hierarchical render pass, with optional temporal conditioning.

    When ``frame_idx`` is provided it is forwarded to both the coarse and
    fine models so that the dust probability (and density/colour) are
    conditioned on the frame/time step.

    Returns a dict with keys:
      coarse/rgb, coarse/depth, coarse/acc, coarse/dust
      fine/rgb,   fine/depth,   fine/acc,   fine/dust
    """
    B = rays_o.shape[0]

    # ---- Coarse pass ----
    pts_c, z_c = sample_stratified(rays_o, rays_d, near, far, n_coarse,
                                   perturb=perturb)
    pts_c_flat  = pts_c.reshape(-1, 3)
    dirs_c_flat = rays_d[:, None, :].expand_as(pts_c).reshape(-1, 3)

    # Expand frame_idx to match flattened points
    fidx_c_flat = None
    if frame_idx is not None:
        fidx_c_flat = frame_idx[:, None].expand(B, n_coarse).reshape(-1)

    sigma_c, rgb_c, dust_c = model_coarse(pts_c_flat, dirs_c_flat, fidx_c_flat)
    sigma_c = sigma_c.view(B, n_coarse)
    rgb_c   = rgb_c.view(B, n_coarse, 3)
    dust_c  = dust_c.view(B, n_coarse)

    rgb_c_map, depth_c, acc_c, weights_c, dust_c_map = volume_render(
        sigma_c, rgb_c, dust_c, z_c, rays_d, white_bkgd)

    # ---- Fine pass (importance sampling) ----
    z_mids = 0.5 * (z_c[:, :-1] + z_c[:, 1:])
    z_fine = sample_pdf(z_mids, weights_c[:, 1:-1].detach(), n_fine,
                        det=(not perturb))
    z_fine = z_fine.detach()

    z_f, _  = torch.sort(torch.cat([z_c, z_fine], dim=-1), dim=-1)
    n_total = z_f.shape[1]

    pts_f       = rays_o[:, None, :] + rays_d[:, None, :] * z_f[..., None]
    pts_f_flat  = pts_f.reshape(-1, 3)
    dirs_f_flat = rays_d[:, None, :].expand_as(pts_f).reshape(-1, 3)

    fidx_f_flat = None
    if frame_idx is not None:
        fidx_f_flat = frame_idx[:, None].expand(B, n_total).reshape(-1)

    sigma_f, rgb_f, dust_f = model_fine(pts_f_flat, dirs_f_flat, fidx_f_flat)
    sigma_f = sigma_f.view(B, n_total)
    rgb_f   = rgb_f.view(B, n_total, 3)
    dust_f  = dust_f.view(B, n_total)

    rgb_f_map, depth_f, acc_f, _, dust_f_map = volume_render(
        sigma_f, rgb_f, dust_f, z_f, rays_d, white_bkgd)

    return {
        "coarse/rgb":   rgb_c_map,
        "coarse/depth": depth_c,
        "coarse/acc":   acc_c,
        "coarse/dust":  dust_c_map,
        "fine/rgb":     rgb_f_map,
        "fine/depth":   depth_f,
        "fine/acc":     acc_f,
        "fine/dust":    dust_f_map,
    }
