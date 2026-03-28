"""
export.py — Export the trained DustNeRF model as a 3-D density grid
             and package everything for download and local visualisation.

Outputs (all under <out_dir>/export/):
  density_grid.npz          — time-averaged 3-D density / dust_prob grid
  temporal_dust_grids.npz   — per-frame dust_prob grids (T × N³), shape
                               (n_frames, res, res, res); also contains
                               frame_times (n_frames,) in seconds
  density_grid.vtk          — VTK legacy ASCII (opens in ParaView / VisIt)
  cameras.json              — camera poses and intrinsics
  dust_cloud.html           — interactive 3-D Plotly figure
  dust_density_slices.png   — Matplotlib slice views (X/Y/Z mid-planes)
  dust_export.tar.gz        — single archive of all of the above

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

from .dataset import load_config, get_camera_entries, get_intrinsics, get_c2w
from .model import DustNeRF


# ---------------------------------------------------------------------------
# Grid evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_density_grid(
    model_fine: DustNeRF,
    scene_min: np.ndarray,
    scene_max: np.ndarray,
    resolution: int,
    batch_size: int = 4096,
    device: torch.device = torch.device("cpu"),
    frame_idx: int = None,
) -> tuple:
    """
    Query the fine model on a regular (resolution³) voxel grid.

    Parameters
    ----------
    frame_idx : int or None
        If given, conditions the model on this temporal frame index.
        Pass None (or any value when n_frames==1) for time-averaged output.

    Returns
    -------
    density   : (res, res, res) float32
    dust_prob : (res, res, res) float32
    """
    xs = np.linspace(scene_min[0], scene_max[0], resolution, dtype=np.float32)
    ys = np.linspace(scene_min[1], scene_max[1], resolution, dtype=np.float32)
    zs = np.linspace(scene_min[2], scene_max[2], resolution, dtype=np.float32)

    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts_np  = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    N       = pts_np.shape[0]
    dirs_np = np.tile([0.0, 0.0, -1.0], (N, 1)).astype(np.float32)

    density_flat   = np.zeros(N, dtype=np.float32)
    dust_prob_flat = np.zeros(N, dtype=np.float32)

    model_fine.eval()

    for i in range(0, N, batch_size):
        pts_b  = torch.from_numpy(pts_np[i:i+batch_size]).to(device)
        dirs_b = torch.from_numpy(dirs_np[i:i+batch_size]).to(device)
        fidx_b = None
        if frame_idx is not None and model_fine.n_frames > 1:
            fidx_b = torch.full(
                (pts_b.shape[0],), frame_idx, dtype=torch.long, device=device)

        sigma, _, dust = model_fine(pts_b, dirs_b, fidx_b)
        density_flat[i:i+batch_size]   = sigma.cpu().numpy()
        dust_prob_flat[i:i+batch_size] = dust.cpu().numpy()

        if (i // batch_size) % 50 == 0:
            print(f"  {100.0 * i / N:.1f}% …", end="\r", flush=True)

    print()
    density   = density_flat.reshape(resolution, resolution, resolution)
    dust_prob = dust_prob_flat.reshape(resolution, resolution, resolution)
    return density, dust_prob


# ---------------------------------------------------------------------------
# Temporal grid export
# ---------------------------------------------------------------------------

@torch.no_grad()
def export_temporal_grids(
    model_fine: DustNeRF,
    scene_min: np.ndarray,
    scene_max: np.ndarray,
    resolution: int,
    n_frames: int,
    frame_times: np.ndarray,
    out_path: str,
    batch_size: int = 4096,
    device: torch.device = torch.device("cpu"),
):
    """
    Evaluate the fine model for every temporal frame index and save the
    resulting per-frame dust probability grids.

    Output file ``temporal_dust_grids.npz`` contains:
      dust_prob_seq  : (n_frames, res, res, res) float32
      density_seq    : (n_frames, res, res, res) float32
      frame_times    : (n_frames,)               float32  [seconds]
      grid_min       : (3,)                       float32
      grid_max       : (3,)                       float32
    """
    if model_fine.n_frames <= 1:
        print("[export] temporal model not active (n_frames=1) — "
              "skipping temporal grid export")
        return

    print(f"[export] generating temporal dust grids: "
          f"{n_frames} frames × {resolution}³ voxels …")

    dust_seq    = np.zeros((n_frames, resolution, resolution, resolution),
                           dtype=np.float32)
    density_seq = np.zeros_like(dust_seq)

    for t in range(n_frames):
        print(f"  frame {t+1}/{n_frames} (t={frame_times[t]:.3f}s) …",
              end=" ", flush=True)
        density_t, dust_t = evaluate_density_grid(
            model_fine, scene_min, scene_max, resolution,
            batch_size=batch_size, device=device, frame_idx=t,
        )
        dust_seq[t]    = dust_t
        density_seq[t] = density_t

    np.savez_compressed(
        out_path,
        dust_prob_seq = dust_seq,
        density_seq   = density_seq,
        frame_times   = frame_times,
        grid_min      = scene_min.astype(np.float32),
        grid_max      = scene_max.astype(np.float32),
    )
    print(f"[export] temporal grids saved → {out_path}")
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"         file size: {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# VTK export
# ---------------------------------------------------------------------------

def save_vtk(path, density, scene_min, scene_max):
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
        vals = density.ravel(order="F")
        chunk = 10
        for s in range(0, len(vals), chunk):
            f.write(" ".join(f"{v:.6f}" for v in vals[s:s+chunk]) + "\n")
    print(f"[export] VTK saved → {path}")


# ---------------------------------------------------------------------------
# Plotly interactive HTML
# ---------------------------------------------------------------------------

def save_plotly_html(path, density, dust_prob, scene_min, scene_max,
                     threshold=0.01):
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

    mask = (density > threshold) & (dust_prob > threshold)
    if not mask.any():
        mask = density > np.percentile(density, 90)

    fig = go.Figure(data=go.Scatter3d(
        x=gx[mask].ravel(), y=gy[mask].ravel(), z=gz[mask].ravel(),
        mode="markers",
        marker=dict(size=2, color=dust_prob[mask].ravel(),
                    colorscale="YlOrRd", opacity=0.6,
                    colorbar=dict(title="Dust prob")),
        text=[f"σ={v:.3f} p={d:.3f}"
              for v, d in zip(density[mask].ravel(),
                              dust_prob[mask].ravel())],
        hoverinfo="text+x+y+z",
    ))
    fig.update_layout(
        title="DustNeRF — 3-D Dust Density Cloud (time-averaged)",
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)",
                   zaxis_title="Z (m)", bgcolor="rgb(10,10,20)"),
        paper_bgcolor="rgb(10,10,20)", font_color="white",
    )
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"[export] Plotly HTML saved → {path}")


# ---------------------------------------------------------------------------
# Matplotlib slice images
# ---------------------------------------------------------------------------

def save_slice_images(path, density, dust_prob):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[export] matplotlib not installed — skipping slice images")
        return

    Nx, Ny, Nz = density.shape
    mx, my, mz = Nx // 2, Ny // 2, Nz // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("DustNeRF — Density Slices (mid-plane, time-averaged)", fontsize=14)

    slices = [
        (density[mx, :, :], dust_prob[mx, :, :], "X-mid"),
        (density[:, my, :], dust_prob[:, my, :], "Y-mid"),
        (density[:, :, mz], dust_prob[:, :, mz], "Z-mid"),
    ]
    for col, (d, p, title) in enumerate(slices):
        axes[0, col].imshow(d.T, origin="lower", cmap="hot",
                            vmin=0, vmax=max(d.max(), 1e-6))
        axes[0, col].set_title(f"{title} — density")
        axes[0, col].axis("off")
        axes[1, col].imshow(p.T, origin="lower", cmap="YlOrRd", vmin=0, vmax=1)
        axes[1, col].set_title(f"{title} — dust prob")
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[export] Slice images saved → {path}")


# ---------------------------------------------------------------------------
# Package
# ---------------------------------------------------------------------------

def package(export_dir, out_archive):
    with tarfile.open(out_archive, "w:gz") as tar:
        tar.add(export_dir, arcname=os.path.basename(export_dir))
    print(f"[export] archive → {out_archive}")


# ---------------------------------------------------------------------------
# Scene-bounds helper (supports both config formats)
# ---------------------------------------------------------------------------

def scene_bounds_from_config(cfg: dict, margin: float = 2.0):
    """Return (scene_min, scene_max) from camera positions in either format."""
    camera_entries = get_camera_entries(cfg)
    positions = []
    for cam in camera_entries:
        c2w = np.array(cam["transform_matrix"], dtype=np.float32)
        positions.append(c2w[:3, 3])
    positions = np.stack(positions, axis=0)
    return positions.min(axis=0) - margin, positions.max(axis=0) + margin


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export(args):
    cfg        = load_config(args.config)
    cfg_train  = cfg["training"]
    cfg_export = cfg.get("export", {})
    resolution       = cfg_export.get("grid_resolution",   128)
    temp_resolution  = cfg_export.get("temporal_resolution", 64)
    threshold        = cfg_export.get("density_threshold", 0.01)
    do_temporal      = cfg_export.get("export_temporal",   True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[export] device = {device}")

    out_dir    = Path(args.out)
    export_dir = out_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    ckpt = torch.load(args.ckpt, map_location=device)
    n_frames = ckpt.get("n_frames", cfg_train.get("n_frames", 1))
    time_dim = cfg_train.get("time_dim", 16)

    fine = DustNeRF(
        pos_freqs=10, dir_freqs=4, net_depth=8, net_width=256,
        n_frames=n_frames, time_dim=time_dim,
    ).to(device)
    fine.load_state_dict(ckpt["fine"])
    print(f"[export] fine model loaded from {args.ckpt} "
          f"(step {ckpt.get('step','?')}, n_frames={n_frames})")

    # ---- Scene bounds ----
    scene_min, scene_max = scene_bounds_from_config(cfg)
    print(f"[export] scene bounds: {scene_min} → {scene_max}")

    # ---- Time-averaged 3D grid ----
    # Use the middle frame for the "canonical" averaged view when n_frames > 1
    mid_frame = (n_frames // 2) if n_frames > 1 else None
    print(f"[export] evaluating time-averaged grid (resolution={resolution}³) …")
    density, dust_prob = evaluate_density_grid(
        fine, scene_min, scene_max, resolution,
        batch_size=8192, device=device, frame_idx=mid_frame,
    )

    # ---- Save NPZ ----
    npz_path = str(export_dir / "density_grid.npz")
    np.savez_compressed(
        npz_path,
        density  = density,
        dust_prob = dust_prob,
        grid_min  = scene_min,
        grid_max  = scene_max,
    )
    print(f"[export] NPZ saved → {npz_path}")

    # ---- Save VTK ----
    save_vtk(str(export_dir / "density_grid.vtk"), density, scene_min, scene_max)

    # ---- Save cameras JSON ----
    cameras_out = []
    for cam in get_camera_entries(cfg):
        cameras_out.append({
            "id": cam["id"],
            "intrinsics": {
                "fl_x": cam["fl_x"], "fl_y": cam["fl_y"],
                "cx":   cam["cx"],   "cy":   cam["cy"],
                "w":    cam["w"],    "h":    cam["h"],
            },
            "c2w":       cam["transform_matrix"],
            "frame_rate": cam["frame_rate"],
            "frame_num":  cam["frame_num"],
        })
    with open(export_dir / "cameras.json", "w") as f:
        json.dump(cameras_out, f, indent=2)
    print("[export] cameras.json saved")

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

    # ---- Per-frame temporal grids ----
    if do_temporal and n_frames > 1:
        # Derive fps from the first camera entry (all cameras are synchronised)
        cam_entries = get_camera_entries(cfg)
        fps         = cam_entries[0]["frame_rate"] if cam_entries else 25.0
        frame_times = np.arange(n_frames, dtype=np.float32) / fps

        export_temporal_grids(
            fine, scene_min, scene_max,
            resolution  = temp_resolution,
            n_frames    = n_frames,
            frame_times = frame_times,
            out_path    = str(export_dir / "temporal_dust_grids.npz"),
            batch_size  = 8192,
            device      = device,
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
    p.add_argument("--config",  default="config/info.json")
    p.add_argument("--ckpt",    required=True)
    p.add_argument("--out",     default="outputs")
    return p.parse_args()


if __name__ == "__main__":
    export(parse_args())
