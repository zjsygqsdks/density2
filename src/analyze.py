"""
analyze.py — Analyze DustNeRF training results and produce an improvement report.

Reads:
  - density_grid.npz      (from export/)
  - train_metrics.jsonl   (from outputs/, downloaded by sync_results.sh)

Produces (in --out directory):
  - analysis_report.json      Human-readable + machine-readable findings
  - training_curves.png       Loss / PSNR curves
  - density_histogram.png     Volume density distribution
  - dust_coverage_map.png     2-D dust coverage projection (top view)

Usage
-----
    python src/analyze.py \\
        --export  results/export/ \\
        --metrics results/train_metrics.jsonl \\
        --out     results/analysis/
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Metric loading
# ---------------------------------------------------------------------------

def load_metrics(jsonl_path: str) -> List[dict]:
    """Load per-step training metrics from a JSONL file."""
    records = []
    if not os.path.isfile(jsonl_path):
        return records
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def load_density_grid(export_dir: str):
    """Load density_grid.npz from export directory."""
    npz_path = os.path.join(export_dir, "density_grid.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"density_grid.npz not found in {export_dir}")
    data = np.load(npz_path)
    return data["density"], data["dust_prob"], data["grid_min"], data["grid_max"]


# ---------------------------------------------------------------------------
# Training curve analysis
# ---------------------------------------------------------------------------

def analyze_training_curves(records: List[dict]) -> dict:
    """
    Detect training issues from per-step metric records.

    Returns a dict with:
      - converged  (bool)
      - final_psnr (float)
      - final_loss (float)
      - diverged   (bool)
      - plateau    (bool)   — PSNR stopped improving in last 20%
      - fast_start (bool)   — loss dropped sharply in first 10%
      - total_steps(int)
      - suggestions: [str]
    """
    if not records:
        return {
            "converged": False,
            "final_psnr": None,
            "final_loss": None,
            "diverged": False,
            "plateau": False,
            "fast_start": False,
            "total_steps": 0,
            "suggestions": ["No training metrics available — cannot assess convergence."],
        }

    steps   = [r["step"]  for r in records]
    losses  = [r["loss"]  for r in records]
    psnrs   = [r["psnr"]  for r in records]
    lrs     = [r["lr"]    for r in records]

    n = len(records)
    final_loss = losses[-1]
    final_psnr = psnrs[-1]
    total_steps = steps[-1]

    suggestions = []

    # Divergence: loss increased monotonically in last 10%
    last10 = losses[int(0.9 * n):]
    diverged = len(last10) >= 3 and all(last10[i] < last10[i+1] for i in range(len(last10)-1))
    if diverged:
        suggestions.append(
            "Training diverged (loss increasing). "
            "Reduce learning_rate by 50% (e.g. 2.5e-4)."
        )

    # Plateau: PSNR improvement < 0.5 dB in last 20% of training
    last20_psnr = psnrs[int(0.8 * n):]
    psnr_range_last20 = max(last20_psnr) - min(last20_psnr)
    plateau = psnr_range_last20 < 0.5 and final_psnr < 28.0
    if plateau and not diverged:
        suggestions.append(
            f"Training plateaued early (PSNR Δ={psnr_range_last20:.2f} dB in last 20%). "
            "Consider increasing max_steps by 50% or reducing lr_decay_factor to 0.05."
        )

    # Low quality: final PSNR < 20 dB
    if final_psnr < 20.0 and not diverged:
        suggestions.append(
            f"Final PSNR is low ({final_psnr:.1f} dB). "
            "Try: (1) increase max_steps, (2) increase n_samples_fine to 192, "
            "(3) increase net_width to 384 in model."
        )

    # High loss early on but stabilised
    first10_loss = losses[:max(1, int(0.1 * n))]
    fast_start = (first10_loss[0] - final_loss) / (first10_loss[0] + 1e-8) > 0.5
    if fast_start:
        suggestions.append(
            "Good fast initial convergence detected. "
            "Consider increasing learning_rate slightly if PSNR is still low."
        )

    converged = not diverged and final_psnr > 18.0
    if converged and final_psnr > 25.0 and not suggestions:
        suggestions.append("Training looks well-converged. No adjustments needed.")

    return {
        "converged":    converged,
        "final_psnr":   round(final_psnr, 2),
        "final_loss":   round(final_loss, 6),
        "diverged":     diverged,
        "plateau":      plateau,
        "fast_start":   fast_start,
        "total_steps":  total_steps,
        "suggestions":  suggestions,
    }


# ---------------------------------------------------------------------------
# Density grid analysis
# ---------------------------------------------------------------------------

def analyze_density_grid(
    density: np.ndarray,
    dust_prob: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
) -> dict:
    """
    Compute spatial statistics on the reconstructed density volume.

    Returns a dict with numerical findings and textual suggestions.
    """
    suggestions = []

    scene_vol = float(np.prod(grid_max - grid_min))
    voxel_vol = scene_vol / density.size

    # Dust coverage
    dust_mask_50 = dust_prob > 0.5
    dust_mask_20 = dust_prob > 0.2
    coverage_50 = float(dust_mask_50.mean())
    coverage_20 = float(dust_mask_20.mean())
    dust_volume_m3 = float(dust_mask_50.sum() * voxel_vol)

    # Density stats
    d_mean  = float(density.mean())
    d_max   = float(density.max())
    d_p95   = float(np.percentile(density, 95))
    d_p99   = float(np.percentile(density, 99))

    # Dust prob stats
    p_mean  = float(dust_prob.mean())

    # Sparsity: fraction of voxels with density < 0.01
    sparsity = float((density < 0.01).mean())

    # Spatial spread — std of dust voxel positions
    if dust_mask_50.any():
        Nx, Ny, Nz = density.shape
        xs = np.linspace(grid_min[0], grid_max[0], Nx)
        ys = np.linspace(grid_min[1], grid_max[1], Ny)
        zs = np.linspace(grid_min[2], grid_max[2], Nz)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        spread_x = float(gx[dust_mask_50].std())
        spread_y = float(gy[dust_mask_50].std())
        spread_z = float(gz[dust_mask_50].std())
    else:
        spread_x = spread_y = spread_z = 0.0

    # Suggestions from grid stats
    if coverage_50 < 0.001:
        suggestions.append(
            f"Dust coverage very low ({coverage_50*100:.3f}% voxels). "
            "Try lowering dust_threshold (e.g. 0.02) and background_frames to 15."
        )
    elif coverage_50 > 0.3:
        suggestions.append(
            f"Dust coverage unexpectedly high ({coverage_50*100:.1f}% voxels — possible background leakage). "
            "Increase dust_threshold (e.g. 0.10) or increase background_frames to 50."
        )

    if sparsity > 0.99:
        suggestions.append(
            "Volume is almost entirely empty. "
            "The model may not have learned dust. Check that data videos actually contain dust."
        )

    if d_max < 0.1:
        suggestions.append(
            f"Maximum density is very low ({d_max:.4f}). "
            "Consider increasing near/far range or n_samples_coarse."
        )

    if p_mean < 0.05:
        suggestions.append(
            f"Average dust_prob is very low ({p_mean:.4f}). "
            "Increase dust_weight_alpha (see config/info.json training.dust_weight_alpha) "
            "or lower dust_threshold."
        )

    # Compute centre of mass of dust cloud
    if dust_mask_50.any():
        cx = float(gx[dust_mask_50].mean())
        cy = float(gy[dust_mask_50].mean())
        cz = float(gz[dust_mask_50].mean())
    else:
        cx = cy = cz = 0.0

    return {
        "coverage_pct_50": round(coverage_50 * 100, 4),
        "coverage_pct_20": round(coverage_20 * 100, 4),
        "dust_volume_m3":  round(dust_volume_m3, 6),
        "density_mean":    round(d_mean, 6),
        "density_max":     round(d_max, 6),
        "density_p95":     round(d_p95, 6),
        "density_p99":     round(d_p99, 6),
        "dust_prob_mean":  round(p_mean, 6),
        "sparsity":        round(sparsity, 6),
        "dust_centre_xyz": [round(cx, 4), round(cy, 4), round(cz, 4)],
        "dust_spread_xyz": [round(spread_x, 4), round(spread_y, 4), round(spread_z, 4)],
        "suggestions":     suggestions,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_training_curves(records: List[dict], out_path: str):
    """Save loss + PSNR curves to a PNG file."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not available — skipping training curves plot")
        return

    if not records:
        return

    steps  = [r["step"]  for r in records]
    losses = [r["loss"]  for r in records]
    psnrs  = [r["psnr"]  for r in records]
    lrs    = [r["lr"]    for r in records]
    loss_dust = [r.get("loss_dust", 0) for r in records]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("DustNeRF Training Curves", fontsize=13)

    axes[0].plot(steps, losses, color="steelblue", linewidth=1.0, label="total loss")
    axes[0].plot(steps, loss_dust, color="orange",   linewidth=0.8, alpha=0.7, label="dust reg")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, psnrs, color="seagreen", linewidth=1.0)
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].axhline(20, color="red",    linestyle="--", linewidth=0.8, label="20 dB")
    axes[1].axhline(25, color="orange", linestyle="--", linewidth=0.8, label="25 dB")
    axes[1].axhline(30, color="green",  linestyle="--", linewidth=0.8, label="30 dB")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, lrs, color="purple", linewidth=1.0)
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_xlabel("Training Step")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[analyze] Training curves → {out_path}")


def plot_density_histogram(density: np.ndarray, dust_prob: np.ndarray, out_path: str):
    """Save density + dust_prob histograms."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("DustNeRF — Volume Distribution", fontsize=12)

    flat_d = density.ravel()
    flat_p = dust_prob.ravel()

    # Only plot non-trivially small values
    nonzero_d = flat_d[flat_d > 0.001]
    ax1.hist(nonzero_d if len(nonzero_d) else flat_d, bins=80,
             color="firebrick", edgecolor="none", alpha=0.8)
    ax1.set_xlabel("Density σ (log scale)")
    ax1.set_ylabel("Voxel count")
    ax1.set_xscale("log")
    ax1.set_title("Density histogram (σ > 0.001)")
    ax1.grid(True, alpha=0.3)

    ax2.hist(flat_p, bins=50, color="darkorange", edgecolor="none", alpha=0.8,
             range=(0, 1))
    ax2.set_xlabel("Dust probability")
    ax2.set_ylabel("Voxel count")
    ax2.set_title("Dust probability distribution")
    ax2.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[analyze] Density histogram → {out_path}")


def plot_dust_coverage_map(
    density: np.ndarray,
    dust_prob: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    out_path: str,
):
    """Top-view (XY) projection of dust probability."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Max-project along Z axis → top view
    top_d = density.max(axis=2)
    top_p = dust_prob.max(axis=2)

    extent = [grid_min[0], grid_max[0], grid_min[1], grid_max[1]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("DustNeRF — Top-View Projection (Z max)", fontsize=12)

    im1 = ax1.imshow(top_d.T, origin="lower", extent=extent, cmap="hot",
                     aspect="auto", vmin=0)
    ax1.set_title("Density σ (Z-max projection)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(top_p.T, origin="lower", extent=extent, cmap="YlOrRd",
                     aspect="auto", vmin=0, vmax=1)
    ax2.set_title("Dust probability (Z-max projection)")
    ax2.set_xlabel("X (m)")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[analyze] Coverage map → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"[analyze] Loading density grid from {args.export} …")
    density, dust_prob, grid_min, grid_max = load_density_grid(args.export)
    print(f"  grid shape : {density.shape}")
    print(f"  scene      : {grid_min} → {grid_max}")

    print(f"[analyze] Loading training metrics from {args.metrics} …")
    records = load_metrics(args.metrics)
    print(f"  metric records : {len(records)}")

    # ── Analyze ───────────────────────────────────────────────────────────────
    training_analysis = analyze_training_curves(records)
    grid_analysis     = analyze_density_grid(density, dust_prob, grid_min, grid_max)

    # ── Combined report ───────────────────────────────────────────────────────
    all_suggestions = training_analysis["suggestions"] + grid_analysis["suggestions"]

    report = {
        "training": training_analysis,
        "density_grid": grid_analysis,
        "all_suggestions": all_suggestions,
    }

    report_path = str(out_dir / "analysis_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[analyze] Report → {report_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training_curves(records, str(out_dir / "training_curves.png"))
    plot_density_histogram(density, dust_prob, str(out_dir / "density_histogram.png"))
    plot_dust_coverage_map(density, dust_prob, grid_min, grid_max,
                           str(out_dir / "dust_coverage_map.png"))

    # ── Print summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Analysis Summary")
    print("=" * 60)
    print(f"  Training steps      : {training_analysis['total_steps']}")
    if training_analysis["final_psnr"] is not None:
        print(f"  Final PSNR          : {training_analysis['final_psnr']:.2f} dB")
        print(f"  Final loss          : {training_analysis['final_loss']:.6f}")
    print(f"  Converged           : {training_analysis['converged']}")
    print(f"  Dust coverage (>50%): {grid_analysis['coverage_pct_50']:.4f}%")
    print(f"  Dust volume         : {grid_analysis['dust_volume_m3']:.4f} m³")
    print()
    if all_suggestions:
        print("  Suggestions:")
        for i, s in enumerate(all_suggestions, 1):
            print(f"    {i}. {s}")
    else:
        print("  No improvement suggestions.")
    print("=" * 60)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyze DustNeRF training results")
    p.add_argument("--export",  default="results/export",
                   help="Path to export directory containing density_grid.npz")
    p.add_argument("--metrics", default="results/train_metrics.jsonl",
                   help="Path to train_metrics.jsonl (may not exist)")
    p.add_argument("--out",     default="results/analysis",
                   help="Output directory for report and plots")
    return p.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
