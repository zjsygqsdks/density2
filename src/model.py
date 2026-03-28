"""
model.py — DustNeRF neural network model.

Architecture
------------
* Positional encoding (Fourier features) for spatial coordinates and directions
* Coarse MLP: (x,y,z) → density σ + intermediate feature
* Fine MLP:   (x,y,z,d_enc) → density σ + RGB colour
* A lightweight "dust head" outputs an extra scalar [0,1] representing
  the probability that a voxel contains dust (as opposed to background).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Fourier-feature positional encoding (Mildenhall et al., NeRF 2020)."""

    def __init__(self, n_freqs: int, include_input: bool = True):
        super().__init__()
        self.n_freqs = n_freqs
        self.include_input = include_input
        freqs = 2.0 ** torch.linspace(0, n_freqs - 1, n_freqs)  # (n_freqs,)
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        """Output dimensionality for a 3-D input."""
        d = 2 * self.n_freqs * 3
        if self.include_input:
            d += 3
        return d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (…, 3)
        returns : (…, out_dim)
        """
        parts = []
        if self.include_input:
            parts.append(x)
        for freq in self.freqs:
            parts.append(torch.sin(freq * x))
            parts.append(torch.cos(freq * x))
        return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# DustNeRF MLP
# ---------------------------------------------------------------------------

class DustNeRF(nn.Module):
    """
    Neural Radiance Field specialised for dust density reconstruction.

    Given a 3-D point p = (x,y,z) and viewing direction d:
    - Outputs σ   (volume density ≥ 0)
    - Outputs rgb (emitted colour ∈ [0,1]³)
    - Outputs dust_prob (probability the point is dust, ∈ [0,1])

    The `dust_prob` head lets us threshold the reconstructed volume to
    extract only the dust cloud (ignoring background structures).
    """

    def __init__(
        self,
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        net_depth: int = 8,
        net_width: int = 256,
        skip_layers: Tuple[int, ...] = (4,),
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(n_freqs=pos_freqs, include_input=True)
        self.dir_enc = PositionalEncoding(n_freqs=dir_freqs, include_input=True)

        pos_dim = self.pos_enc.out_dim   # 3 + 2*10*3 = 63
        dir_dim = self.dir_enc.out_dim   # 3 + 2* 4*3 = 27

        self.skip_layers = skip_layers

        # Build position MLP (σ + feature)
        layers = []
        in_dim = pos_dim
        for i in range(net_depth):
            layers.append(nn.Linear(in_dim, net_width))
            in_dim = net_width
            if i in skip_layers:
                in_dim += pos_dim          # residual skip connection

        self.pos_layers = nn.ModuleList(layers)

        # Density head
        self.sigma_head = nn.Linear(net_width, 1)
        # Dust probability head (separate branch)
        self.dust_head = nn.Linear(net_width, 1)
        # Feature vector for colour branch
        self.feature_head = nn.Linear(net_width, net_width)

        # Colour MLP
        self.colour_layer1 = nn.Linear(net_width + dir_dim, net_width // 2)
        self.colour_layer2 = nn.Linear(net_width // 2, 3)

    # ------------------------------------------------------------------

    def forward(
        self,
        pts: torch.Tensor,          # (N, 3)
        dirs: torch.Tensor,         # (N, 3)  unit direction vectors
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        sigma     : (N,)   volume density (non-negative)
        rgb       : (N, 3) emitted colour [0,1]
        dust_prob : (N,)   dust probability [0,1]
        """
        pos_enc = self.pos_enc(pts)   # (N, pos_dim)
        dir_enc = self.dir_enc(dirs)  # (N, dir_dim)

        h = pos_enc
        for i, layer in enumerate(self.pos_layers):
            if i in self.skip_layers:
                h = torch.cat([h, pos_enc], dim=-1)
            h = F.relu(layer(h))

        # Sigma (density)
        sigma = F.softplus(self.sigma_head(h)).squeeze(-1)   # (N,)  ≥ 0

        # Dust probability
        dust_prob = torch.sigmoid(self.dust_head(h)).squeeze(-1)  # (N,) ∈[0,1]

        # Colour
        feat = self.feature_head(h)                                # (N, W)
        colour_in = torch.cat([feat, dir_enc], dim=-1)
        colour_h = F.relu(self.colour_layer1(colour_in))
        rgb = torch.sigmoid(self.colour_layer2(colour_h))         # (N, 3) ∈[0,1]

        return sigma, rgb, dust_prob


# ---------------------------------------------------------------------------
# Hierarchical sampling model (coarse + fine)
# ---------------------------------------------------------------------------

class HierarchicalDustNeRF(nn.Module):
    """
    Two-level (coarse + fine) DustNeRF for importance sampling.
    """

    def __init__(self, coarse_kwargs: dict = None, fine_kwargs: dict = None):
        super().__init__()
        coarse_kwargs = coarse_kwargs or {}
        fine_kwargs = fine_kwargs or {}
        self.coarse = DustNeRF(**coarse_kwargs)
        self.fine = DustNeRF(**fine_kwargs)
