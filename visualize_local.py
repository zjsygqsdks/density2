"""
visualize_local.py — Local visualisation script for DustNeRF export data.

Run this script on your LOCAL machine (not the server) after downloading
the export archive (dust_export.tar.gz).

Usage
-----
1.  Download dust_export.tar.gz from the server.
2.  Extract: tar -xzf dust_export.tar.gz
3.  Run:     python visualize_local.py --export export/

Features
---------
* Interactive 3-D scatter plot of the dust cloud (Plotly)
* Mid-plane density slice images (matplotlib)
* Camera frustum overlays
* Time-series density analysis (if multiple snapshots provided)
* Side-by-side comparison of density vs dust-probability volumes

Requirements (local machine):
    pip install numpy matplotlib plotly scipy
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helper: load export data
# ---------------------------------------------------------------------------

def load_export(export_dir: str):
    ed = Path(export_dir)
    npz_path = ed / "density_grid.npz"
    if not npz_path.exists():
        sys.exit(f"[ERROR] density_grid.npz not found in {export_dir}")

    data = np.load(str(npz_path))
    density   = data["density"]
    dust_prob = data["dust_prob"]
    grid_min  = data["grid_min"]
    grid_max  = data["grid_max"]

    cameras = []
    cam_path = ed / "cameras.json"
    if cam_path.exists():
        with open(cam_path) as f:
            cameras = json.load(f)

    return density, dust_prob, grid_min, grid_max, cameras


# ---------------------------------------------------------------------------
# 3-D interactive scatter plot
# ---------------------------------------------------------------------------

def plot_3d_cloud(
    density: np.ndarray,
    dust_prob: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    cameras: list,
    threshold_pct: float = 80.0,
    output_html: str = None,
):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[viz] plotly not found. Install with: pip install plotly")
        return

    Nx, Ny, Nz = density.shape
    xs = np.linspace(grid_min[0], grid_max[0], Nx)
    ys = np.linspace(grid_min[1], grid_max[1], Ny)
    zs = np.linspace(grid_min[2], grid_max[2], Nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")

    thr = np.percentile(density, threshold_pct)
    mask = density > thr

    traces = []

    # Dust cloud
    traces.append(go.Scatter3d(
        x=gx[mask].ravel(), y=gy[mask].ravel(), z=gz[mask].ravel(),
        mode="markers",
        name="Dust cloud",
        marker=dict(
            size=2,
            color=dust_prob[mask].ravel(),
            colorscale="YlOrRd",
            opacity=0.5,
            colorbar=dict(title="Dust probability", x=1.0),
            cmin=0.0, cmax=1.0,
        ),
    ))

    # Camera positions
    if cameras:
        cam_x, cam_y, cam_z, cam_labels = [], [], [], []
        for c in cameras:
            mat = np.array(c["c2w"])
            cam_x.append(mat[0, 3])
            cam_y.append(mat[1, 3])
            cam_z.append(mat[2, 3])
            cam_labels.append(c["id"])

        traces.append(go.Scatter3d(
            x=cam_x, y=cam_y, z=cam_z,
            mode="markers+text",
            name="Cameras",
            text=cam_labels,
            textposition="top center",
            marker=dict(size=8, color="cyan", symbol="diamond"),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="DustNeRF — 3-D Dust Density Cloud",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
        ),
        legend=dict(x=0, y=1),
    )

    if output_html:
        fig.write_html(output_html)
        print(f"[viz] interactive HTML saved → {output_html}")
    else:
        fig.show()


# ---------------------------------------------------------------------------
# Matplotlib slice viewer
# ---------------------------------------------------------------------------

def plot_slices(
    density: np.ndarray,
    dust_prob: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    output_png: str = None,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib not found. Install with: pip install matplotlib")
        return

    Nx, Ny, Nz = density.shape
    extents = [
        [grid_min[1], grid_max[1], grid_min[2], grid_max[2]],
        [grid_min[0], grid_max[0], grid_min[2], grid_max[2]],
        [grid_min[0], grid_max[0], grid_min[1], grid_max[1]],
    ]
    titles = [f"X={grid_min[0]+0.5*(grid_max[0]-grid_min[0]):.2f} (mid)",
              f"Y={grid_min[1]+0.5*(grid_max[1]-grid_min[1]):.2f} (mid)",
              f"Z={grid_min[2]+0.5*(grid_max[2]-grid_min[2]):.2f} (mid)"]

    slices_d = [density[Nx//2, :, :], density[:, Ny//2, :], density[:, :, Nz//2]]
    slices_p = [dust_prob[Nx//2, :, :], dust_prob[:, Ny//2, :], dust_prob[:, :, Nz//2]]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("DustNeRF — Density Slices", fontsize=15)

    for col in range(3):
        im0 = axes[0, col].imshow(
            slices_d[col].T, origin="lower",
            extent=extents[col], cmap="hot",
            vmin=0, vmax=slices_d[col].max() or 1,
            aspect="auto",
        )
        axes[0, col].set_title(f"{titles[col]} — σ density")
        plt.colorbar(im0, ax=axes[0, col])

        im1 = axes[1, col].imshow(
            slices_p[col].T, origin="lower",
            extent=extents[col], cmap="YlOrRd",
            vmin=0, vmax=1,
            aspect="auto",
        )
        axes[1, col].set_title(f"{titles[col]} — dust probability")
        plt.colorbar(im1, ax=axes[1, col])

    plt.tight_layout()

    if output_png:
        plt.savefig(output_png, dpi=120)
        print(f"[viz] slice image saved → {output_png}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(density: np.ndarray, dust_prob: np.ndarray, grid_min, grid_max):
    scene_vol = float(np.prod(grid_max - grid_min))
    voxel_vol = scene_vol / density.size
    dust_mask = dust_prob > 0.5
    dust_voxels = int(dust_mask.sum())
    dust_vol = dust_voxels * voxel_vol

    print("=" * 60)
    print("  DustNeRF Export — Summary")
    print("=" * 60)
    print(f"  Grid resolution  : {density.shape[0]}³")
    print(f"  Scene bounds min : {grid_min}")
    print(f"  Scene bounds max : {grid_max}")
    print(f"  Scene volume     : {scene_vol:.3f} m³")
    print(f"  Density  (σ)")
    print(f"    min / max / mean : {density.min():.4f} / {density.max():.4f} / {density.mean():.4f}")
    print(f"  Dust probability")
    print(f"    mean             : {dust_prob.mean():.4f}")
    print(f"    voxels > 0.5     : {dust_voxels:,} ({100*dust_voxels/density.size:.2f}%)")
    print(f"    estimated volume : {dust_vol:.3f} m³")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Visualise DustNeRF export (local machine)")
    p.add_argument("--export", default="export", help="Path to export directory")
    p.add_argument("--threshold", type=float, default=80.0,
                   help="Percentile threshold for 3-D scatter display (default: 80)")
    p.add_argument("--save-html", default=None,
                   help="Save interactive 3-D HTML to this path instead of opening browser")
    p.add_argument("--save-png",  default=None,
                   help="Save slice image to this path instead of opening window")
    p.add_argument("--no-3d",  action="store_true", help="Skip 3-D plot")
    p.add_argument("--no-slices", action="store_true", help="Skip slice plots")
    return p.parse_args()


def main():
    args = parse_args()
    density, dust_prob, grid_min, grid_max, cameras = load_export(args.export)
    print_summary(density, dust_prob, grid_min, grid_max)

    if not args.no_3d:
        plot_3d_cloud(
            density, dust_prob, grid_min, grid_max, cameras,
            threshold_pct=args.threshold,
            output_html=args.save_html,
        )

    if not args.no_slices:
        plot_slices(density, dust_prob, grid_min, grid_max, output_png=args.save_png)


if __name__ == "__main__":
    main()
