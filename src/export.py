"""
export.py — Export the trained DustNeRF model as a 3-D density grid
             and package everything for download and local visualisation.

Outputs (all under <out_dir>/export/):
  density_grid.npz        — NumPy compressed archive with:
                             • density  (Nx×Ny×Nz float32)
                             • dust_prob(Nx×Ny×Nz float32)
                             • grid_min / grid_max (3,) scene bounds
  density_grid.vtk        — VTK legacy ASCII file (opens in ParaView / VisIt)
  cameras.json            — Camera poses and intrinsics (for viz script)
  dust_cloud.html         — Interactive 3-D Plotly figure (open in browser)
  dust_density_slices.png — Matplotlib slice views (X/Y/Z mid-planes)
  export.tar.gz           — Single archive containing all of the above

Usage:
    python -m src.export --config config/info.json \\
                         --ckpt   outputs/checkpoints/ckpt_latest.pt \\
                         --out    outputs/
"""

import argparse
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import torch

from .dataset import load_config, get_intrinsics, get_c2w
from .model import DustNeRF


# ---------------------------------------------------------------------------
# Grid evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_density_grid(
    model_fine: DustNeRF,
    scene_min: np.ndarray,
    scene_max: np.ndarray,
    resolution: int,
    batch_size: int = 4096,
    device: torch.device = torch.device("cpu"),
) -> tuple:
    """
    Query the fine model on a regular (resolution³) voxel grid.

    Returns
    -------
    density   : (Nx, Ny, Nz) float32
    dust_prob : (Nx, Ny, Nz) float32
    """
    xs = np.linspace(scene_min[0], scene_max[0], resolution, dtype=np.float32)
    ys = np.linspace(scene_min[1], scene_max[1], resolution, dtype=np.float32)
    zs = np.linspace(scene_min[2], scene_max[2], resolution, dtype=np.float32)

    # meshgrid → flat list of points
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")   # each (Nx, Ny, Nz)
    pts_np = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)  # (N, 3)
    N = pts_np.shape[0]

    # dummy view direction (pointing down -z, same for all query points)
    dirs_np = np.tile([0.0, 0.0, -1.0], (N, 1)).astype(np.float32)

    density_flat   = np.zeros(N, dtype=np.float32)
    dust_prob_flat = np.zeros(N, dtype=np.float32)

    model_fine.eval()
    print(f"[export] evaluating {N:,} grid points (resolution={resolution}³) …")

    for i in range(0, N, batch_size):
        pts_b  = torch.from_numpy(pts_np[i:i+batch_size]).to(device)
        dirs_b = torch.from_numpy(dirs_np[i:i+batch_size]).to(device)
        sigma, _, dust = model_fine(pts_b, dirs_b)
        density_flat[i:i+batch_size]   = sigma.cpu().numpy()
        dust_prob_flat[i:i+batch_size] = dust.cpu().numpy()

        if (i // batch_size) % 50 == 0:
            pct = 100.0 * i / N
            print(f"  {pct:.1f}% …", end="\r", flush=True)

    print()
    density   = density_flat.reshape(resolution, resolution, resolution)
    dust_prob = dust_prob_flat.reshape(resolution, resolution, resolution)
    return density, dust_prob


# ---------------------------------------------------------------------------
# VTK export (legacy ASCII, readable by ParaView / VisIt without libraries)
# ---------------------------------------------------------------------------

def save_vtk(
    path: str,
    density: np.ndarray,
    scene_min: np.ndarray,
    scene_max: np.ndarray,
):
    Nx, Ny, Nz = density.shape
    dx = (scene_max[0] - scene_min[0]) / Nx
    dy = (scene_max[1] - scene_min[1]) / Ny
    dz = (scene_max[2] - scene_min[2]) / Nz

    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("DustNeRF density grid\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {Nx} {Ny} {Nz}\n")
        f.write(f"ORIGIN {scene_min[0]:.6f} {scene_min[1]:.6f} {scene_min[2]:.6f}\n")
        f.write(f"SPACING {dx:.6f} {dy:.6f} {dz:.6f}\n")
        f.write(f"POINT_DATA {Nx * Ny * Nz}\n")
        f.write("SCALARS density float 1\n")
        f.write("LOOKUP_TABLE default\n")
        vals = density.ravel(order="F")   # Fortran order for VTK
        chunk_size = 10
        for start in range(0, len(vals), chunk_size):
            line = " ".join(f"{v:.6f}" for v in vals[start:start+chunk_size])
            f.write(line + "\n")
    print(f"[export] VTK saved → {path}")


# ---------------------------------------------------------------------------
# Plotly interactive HTML
# ---------------------------------------------------------------------------

def save_plotly_html(
    path: str,
    density: np.ndarray,
    dust_prob: np.ndarray,
    scene_min: np.ndarray,
    scene_max: np.ndarray,
    threshold: float = 0.01,
):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[export] plotly not installed — skipping HTML export")
        return

    Nx, Ny, Nz = density.shape
    xs = np.linspace(scene_min[0], scene_max[0], Nx)
    ys = np.linspace(scene_min[1], scene_max[1], Ny)
    zs = np.linspace(scene_min[2], scene_max[2], Nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")

    # Show only voxels above threshold (sparse cloud)
    mask = (density > threshold) & (dust_prob > threshold)
    if not mask.any():
        threshold_adj = np.percentile(density, 90)
        mask = density > threshold_adj

    px = gx[mask].ravel()
    py = gy[mask].ravel()
    pz = gz[mask].ravel()
    pv = density[mask].ravel()
    pd = dust_prob[mask].ravel()

    fig = go.Figure(data=go.Scatter3d(
        x=px, y=py, z=pz,
        mode="markers",
        marker=dict(
            size=2,
            color=pd,
            colorscale="YlOrRd",
            opacity=0.6,
            colorbar=dict(title="Dust prob"),
        ),
        text=[f"σ={v:.3f} p={d:.3f}" for v, d in zip(pv, pd)],
        hoverinfo="text+x+y+z",
    ))
    fig.update_layout(
        title="DustNeRF — 3-D Dust Density Cloud",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            bgcolor="rgb(10,10,20)",
        ),
        paper_bgcolor="rgb(10,10,20)",
        font_color="white",
    )
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"[export] Plotly HTML saved → {path}")


# ---------------------------------------------------------------------------
# Matplotlib slice images
# ---------------------------------------------------------------------------

def save_slice_images(
    path: str,
    density: np.ndarray,
    dust_prob: np.ndarray,
):
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless back-end
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print("[export] matplotlib not installed — skipping slice images")
        return

    Nx, Ny, Nz = density.shape
    mid_x, mid_y, mid_z = Nx // 2, Ny // 2, Nz // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("DustNeRF — Density Slices (mid-plane)", fontsize=14)

    slices = [
        (density[mid_x, :, :],   dust_prob[mid_x, :, :],   "X-mid slice"),
        (density[:, mid_y, :],   dust_prob[:, mid_y, :],   "Y-mid slice"),
        (density[:, :, mid_z],   dust_prob[:, :, mid_z],   "Z-mid slice"),
    ]
    for col, (d_slice, p_slice, title) in enumerate(slices):
        vmax_d = d_slice.max() if d_slice.max() > 0 else 1.0
        axes[0, col].imshow(d_slice.T, origin="lower", cmap="hot", vmin=0, vmax=vmax_d)
        axes[0, col].set_title(f"{title} — density")
        axes[0, col].axis("off")

        axes[1, col].imshow(p_slice.T, origin="lower", cmap="YlOrRd", vmin=0, vmax=1)
        axes[1, col].set_title(f"{title} — dust prob")
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[export] Slice images saved → {path}")


# ---------------------------------------------------------------------------
# Package everything into a tar.gz
# ---------------------------------------------------------------------------

def package(export_dir: str, out_archive: str):
    with tarfile.open(out_archive, "w:gz") as tar:
        tar.add(export_dir, arcname=os.path.basename(export_dir))
    print(f"[export] archive → {out_archive}")


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export(args):
    cfg = load_config(args.config)
    cfg_train  = cfg["training"]
    cfg_export = cfg.get("export", {})
    resolution = cfg_export.get("grid_resolution", 128)
    threshold  = cfg_export.get("density_threshold", 0.01)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[export] device = {device}")

    out_dir = Path(args.out)
    export_dir = out_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    fine = DustNeRF(pos_freqs=10, dir_freqs=4, net_depth=8, net_width=256).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    fine.load_state_dict(ckpt["fine"])
    print(f"[export] fine model loaded from {args.ckpt} (step {ckpt.get('step', '?')})")

    # ---- Scene bounds from camera positions ----
    positions = []
    for cam_cfg in cfg["cameras"]:
        c2w = np.array(cam_cfg["transform_matrix"], dtype=np.float32)
        positions.append(c2w[:3, 3])
    positions = np.stack(positions, axis=0)
    margin = 2.0
    scene_min = positions.min(axis=0) - margin
    scene_max = positions.max(axis=0) + margin
    print(f"[export] scene bounds: {scene_min} → {scene_max}")

    # ---- Evaluate grid ----
    density, dust_prob = evaluate_density_grid(
        fine, scene_min, scene_max, resolution,
        batch_size=8192, device=device,
    )

    # ---- Save NPZ ----
    npz_path = str(export_dir / "density_grid.npz")
    np.savez_compressed(
        npz_path,
        density=density,
        dust_prob=dust_prob,
        grid_min=scene_min,
        grid_max=scene_max,
    )
    print(f"[export] NPZ saved → {npz_path}")

    # ---- Save VTK ----
    save_vtk(str(export_dir / "density_grid.vtk"), density, scene_min, scene_max)

    # ---- Save cameras JSON ----
    cameras_out = []
    for cam_cfg in cfg["cameras"]:
        cameras_out.append({
            "id": cam_cfg["id"],
            "intrinsics": {
                "fl_x": cam_cfg["fl_x"], "fl_y": cam_cfg["fl_y"],
                "cx": cam_cfg["cx"],     "cy": cam_cfg["cy"],
                "w": cam_cfg["w"],       "h": cam_cfg["h"],
            },
            "c2w": cam_cfg["transform_matrix"],
        })
    with open(export_dir / "cameras.json", "w") as f:
        json.dump(cameras_out, f, indent=2)
    print(f"[export] cameras.json saved")

    # ---- Plotly HTML ----
    save_plotly_html(
        str(export_dir / "dust_cloud.html"),
        density, dust_prob, scene_min, scene_max, threshold=threshold,
    )

    # ---- Matplotlib slices ----
    save_slice_images(
        str(export_dir / "dust_density_slices.png"),
        density, dust_prob,
    )

    # ---- Package ----
    archive_path = str(out_dir / "dust_export.tar.gz")
    package(str(export_dir), archive_path)

    print(f"\n[export] All files are in: {export_dir}")
    print(f"[export] Download archive: {archive_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Export DustNeRF density grid")
    p.add_argument("--config", default="config/info.json")
    p.add_argument("--ckpt",   required=True, help="Path to checkpoint .pt file")
    p.add_argument("--out",    default="outputs")
    return p.parse_args()


if __name__ == "__main__":
    export(parse_args())
