"""
model.py — DustNeRF neural network model.

Architecture
------------
* Positional encoding (Fourier features) for spatial coordinates and directions
* Coarse MLP: (x,y,z) → density σ + intermediate feature
* Fine MLP:   (x,y,z,d_enc) → density σ + RGB colour
* A lightweight "dust head" outputs an extra scalar [0,1] representing
  the probability that a voxel contains dust (as opposed to background).

Temporal extension
------------------
When ``n_frames > 1`` the model is also conditioned on a **frame index**
(integer in [0, n_frames)).  A learnable embedding of dimension ``time_dim``
is injected into the shared MLP feature vector so that the density σ, colour
rgb, and dust probability all vary with the captured time step.  This enables
per-frame dust distribution reconstruction from fixed-pose cameras.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Fourier-feature positional encoding (Mildenhall et al., NeRF 2020)."""

    def __init__(self, n_freqs: int, include_input: bool = True):
        super().__init__()
        self.n_freqs = n_freqs
        self.include_input = include_input
        freqs = 2.0 ** torch.linspace(0, n_freqs - 1, n_freqs)
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        d = 2 * self.n_freqs * 3
        if self.include_input:
            d += 3
        return d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (…, 3)  →  (…, out_dim)"""
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

    Given a 3-D point p = (x,y,z), viewing direction d, and an optional
    frame index t:
    - Outputs σ         (volume density ≥ 0)
    - Outputs rgb       (emitted colour ∈ [0,1]³)
    - Outputs dust_prob (probability the point is dust, ∈ [0,1])

    When ``n_frames > 1``, a learnable time embedding of dimension
    ``time_dim`` is concatenated with the shared position-MLP features
    and projected back to ``net_width`` via a single linear layer.  This
    allows all three outputs to vary with the frame (time step), enabling
    temporal dust reconstruction with static camera poses.
    """

    def __init__(
        self,
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        net_depth: int = 8,
        net_width: int = 256,
        skip_layers: Tuple[int, ...] = (4,),
        n_frames: int = 1,
        time_dim: int = 16,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(n_freqs=pos_freqs, include_input=True)
        self.dir_enc = PositionalEncoding(n_freqs=dir_freqs, include_input=True)

        pos_dim = self.pos_enc.out_dim   # 3 + 2*10*3 = 63
        dir_dim = self.dir_enc.out_dim   # 3 + 2* 4*3 = 27

        self.skip_layers = skip_layers
        self.n_frames = n_frames
        self.time_dim = time_dim

        # Build position MLP (σ + feature)
        layers = []
        in_dim = pos_dim
        for i in range(net_depth):
            # Apply skip by widening input BEFORE creating the layer so that
            # layer[i] is Linear(net_width + pos_dim, net_width) when i is a
            # skip layer.  The forward pass concatenates pos_enc to h right
            # before passing it through this layer.
            if i in skip_layers:
                in_dim += pos_dim
            layers.append(nn.Linear(in_dim, net_width))
            in_dim = net_width
        self.pos_layers = nn.ModuleList(layers)

        # ---- Temporal conditioning (only when n_frames > 1) ----
        if n_frames > 1:
            self.time_emb = nn.Embedding(n_frames, time_dim)
            # Project (net_width + time_dim) back to net_width
            self.time_inject = nn.Linear(net_width + time_dim, net_width)
        else:
            self.time_emb    = None
            self.time_inject = None

        # Density head
        self.sigma_head = nn.Linear(net_width, 1)
        # Dust probability head (separate branch, also sees time)
        self.dust_head = nn.Linear(net_width, 1)
        # Feature vector for colour branch
        self.feature_head = nn.Linear(net_width, net_width)

        # Colour MLP
        self.colour_layer1 = nn.Linear(net_width + dir_dim, net_width // 2)
        self.colour_layer2 = nn.Linear(net_width // 2, 3)

    # ------------------------------------------------------------------

    def forward(
        self,
        pts: torch.Tensor,                       # (N, 3)
        dirs: torch.Tensor,                      # (N, 3)  unit directions
        frame_idx: Optional[torch.Tensor] = None,# (N,)    int64 frame index
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

        # ---- Inject time embedding (if temporal model) ----
        if self.time_emb is not None and frame_idx is not None:
            t_emb = self.time_emb(frame_idx)                    # (N, time_dim)
            h = F.relu(self.time_inject(torch.cat([h, t_emb], dim=-1)))  # (N, W)

        # Sigma (density)
        sigma = F.softplus(self.sigma_head(h)).squeeze(-1)      # (N,)  ≥ 0

        # Dust probability
        dust_prob = torch.sigmoid(self.dust_head(h)).squeeze(-1)  # (N,) ∈[0,1]

        # Colour
        feat = self.feature_head(h)
        colour_in = torch.cat([feat, dir_enc], dim=-1)
        colour_h = F.relu(self.colour_layer1(colour_in))
        rgb = torch.sigmoid(self.colour_layer2(colour_h))       # (N, 3) ∈[0,1]

        return sigma, rgb, dust_prob


# ---------------------------------------------------------------------------
# Hierarchical sampling model (coarse + fine)
# ---------------------------------------------------------------------------

class HierarchicalDustNeRF(nn.Module):
    """Two-level (coarse + fine) DustNeRF for importance sampling."""

    def __init__(self, coarse_kwargs: dict = None, fine_kwargs: dict = None):
        super().__init__()
        coarse_kwargs = coarse_kwargs or {}
        fine_kwargs   = fine_kwargs   or {}
        self.coarse = DustNeRF(**coarse_kwargs)
        self.fine   = DustNeRF(**fine_kwargs)
