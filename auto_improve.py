#!/usr/bin/env python3
"""
auto_improve.py — Read an analysis_report.json and automatically patch
                  config/info.json with improved hyperparameters.

Usage
-----
    python auto_improve.py \\
        --report results/analysis/analysis_report.json \\
        --config config/info.json \\
        [--dry-run]     # print changes without writing

The script:
  1. Reads the structured suggestions from analysis_report.json
  2. Applies a deterministic rule set to derive new hyperparameter values
  3. Writes a backup of the original config as config/info.json.bak
  4. Writes the updated config/info.json
  5. Prints a human-readable diff of all changed values
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Rule constants  (tune these to adjust the auto-improvement behaviour)
# ---------------------------------------------------------------------------

LR_DIVERGE_FACTOR      = 0.5       # multiply LR by this when diverged
STEPS_LOW_PSNR_FACTOR  = 1.5       # multiply max_steps by this when PSNR < LOW_PSNR_THRESHOLD
STEPS_PLATEAU_FACTOR   = 1.3       # multiply max_steps by this on plateau
LR_DECAY_SLOW          = 0.05      # lr_decay_factor to use on plateau
LOW_PSNR_THRESHOLD     = 20.0      # dB — below this → add more training steps
PSNR_HIGH_THRESHOLD    = 25.0      # dB — above this → use higher export resolution
DUST_ALPHA_FACTOR      = 2.0       # multiply dust_weight_alpha by this when p_mean < DUST_PROB_LOW
DUST_PROB_LOW          = 0.05      # threshold for "dust not learned"
COVERAGE_TOO_LOW       = 0.01      # % — coverage below this → lower dust_threshold
COVERAGE_TOO_HIGH      = 30.0      # % — coverage above this → raise dust_threshold
DUST_THRESHOLD_LOW_FACTOR  = 0.5   # multiply dust_threshold when coverage too low
DUST_THRESHOLD_HIGH_FACTOR = 1.5   # multiply dust_threshold when coverage too high
EXPORT_RES_HIGH        = 256       # grid_resolution when PSNR is high
SAMPLES_SPARSE_FACTOR  = 2         # multiply n_samples when volume near-empty
SPARSITY_THRESHOLD     = 0.995     # fraction of near-zero voxels → "near-empty"

# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------

def apply_rules(
    cfg: dict,
    report: dict,
) -> Tuple[dict, List[str]]:
    """
    Apply improvement rules based on analysis report findings.

    Returns
    -------
    new_cfg   : updated config dict (deep copy)
    changelog : list of human-readable change descriptions
    """
    new_cfg = copy.deepcopy(cfg)
    t = new_cfg["training"]
    changelog: List[str] = []

    training = report.get("training", {})
    grid     = report.get("density_grid", {})

    def change(key: str, old_val: Any, new_val: Any, reason: str):
        if old_val != new_val:
            t[key] = new_val
            changelog.append(
                f"  training.{key}: {old_val} → {new_val}  ({reason})"
            )

    # ── 1. Divergence → reduce learning rate ─────────────────────────────────
    if training.get("diverged", False):
        old_lr = t.get("learning_rate", 5e-4)
        change("learning_rate", old_lr, round(old_lr * LR_DIVERGE_FACTOR, 6),
               "training diverged")

    # ── 2. Low PSNR → more training steps ────────────────────────────────────
    final_psnr = training.get("final_psnr")
    if final_psnr is not None and final_psnr < LOW_PSNR_THRESHOLD \
            and not training.get("diverged", False):
        old_steps = t.get("max_steps", 200000)
        new_steps = int(old_steps * STEPS_LOW_PSNR_FACTOR)
        change("max_steps", old_steps, new_steps,
               f"PSNR={final_psnr:.1f}dB < {LOW_PSNR_THRESHOLD} dB")

    # ── 3. Early plateau → more steps + slower LR decay ──────────────────────
    if training.get("plateau", False) and not training.get("diverged", False):
        old_steps = t.get("max_steps", 200000)
        new_steps = max(old_steps, int(old_steps * STEPS_PLATEAU_FACTOR))
        change("max_steps", old_steps, new_steps, "training plateau detected")

        old_decay_factor = t.get("lr_decay_factor", 0.1)
        if old_decay_factor > LR_DECAY_SLOW:
            change("lr_decay_factor", old_decay_factor, LR_DECAY_SLOW,
                   "plateau — slower LR decay")

    # ── 4. Low dust coverage → lower threshold ────────────────────────────────
    coverage = grid.get("coverage_pct_50", None)
    if coverage is not None:
        if coverage < COVERAGE_TOO_LOW:
            old_thr = t.get("dust_threshold", 0.05)
            new_thr = max(0.01, round(old_thr * DUST_THRESHOLD_LOW_FACTOR, 3))
            change("dust_threshold", old_thr, new_thr,
                   f"dust coverage very low ({coverage:.4f}%)")

            old_bg = t.get("background_frames", 30)
            new_bg = max(10, old_bg - 10)
            change("background_frames", old_bg, new_bg,
                   "reduce BG frames to increase sensitivity")

        elif coverage > COVERAGE_TOO_HIGH:
            # Background leakage
            old_thr = t.get("dust_threshold", 0.05)
            new_thr = min(0.20, round(old_thr * DUST_THRESHOLD_HIGH_FACTOR, 3))
            change("dust_threshold", old_thr, new_thr,
                   f"dust coverage too high ({coverage:.1f}% — possible BG leakage)")

            old_bg = t.get("background_frames", 30)
            new_bg = min(60, old_bg + 15)
            change("background_frames", old_bg, new_bg,
                   "more BG frames to improve BG estimation")

    # ── 5. Very sparse volume (almost no density) ─────────────────────────────
    sparsity = grid.get("sparsity", None)
    if sparsity is not None and sparsity > SPARSITY_THRESHOLD:
        old_coarse = t.get("n_samples_coarse", 64)
        new_coarse = min(128, old_coarse * SAMPLES_SPARSE_FACTOR)
        change("n_samples_coarse", old_coarse, new_coarse,
               "volume near-empty — denser coarse sampling")

        old_fine = t.get("n_samples_fine", 128)
        new_fine = min(256, old_fine * SAMPLES_SPARSE_FACTOR)
        change("n_samples_fine", old_fine, new_fine,
               "volume near-empty — denser fine sampling")

    # ── 6. Low dust prob mean → increase dust weight in loss ─────────────────
    p_mean = grid.get("dust_prob_mean", None)
    if p_mean is not None and p_mean < DUST_PROB_LOW:
        old_dwa = t.get("dust_weight_alpha", 5.0)
        new_dwa = min(15.0, old_dwa * DUST_ALPHA_FACTOR)
        change("dust_weight_alpha", old_dwa, new_dwa,
               f"dust_prob_mean={p_mean:.4f} too low — increase dust weight")

    # ── 7. Good results: optionally suggest higher resolution export ──────────
    if final_psnr is not None and final_psnr >= PSNR_HIGH_THRESHOLD \
            and not training.get("diverged", False):
        old_res = new_cfg.get("export", {}).get("grid_resolution", 128)
        if old_res < EXPORT_RES_HIGH:
            new_cfg.setdefault("export", {})["grid_resolution"] = EXPORT_RES_HIGH
            changelog.append(
                f"  export.grid_resolution: {old_res} → {EXPORT_RES_HIGH}  "
                f"(PSNR >= {PSNR_HIGH_THRESHOLD} dB — higher resolution export)"
            )

    return new_cfg, changelog


# ---------------------------------------------------------------------------
# Diff printer
# ---------------------------------------------------------------------------

def dict_diff(old: dict, new: dict, path: str = "") -> List[str]:
    """Recursively compute changed key-value pairs between two dicts."""
    diffs = []
    all_keys = set(old) | set(new)
    for k in sorted(all_keys):
        full_key = f"{path}.{k}" if path else k
        if k not in old:
            diffs.append(f"  + {full_key}: (added) {new[k]}")
        elif k not in new:
            diffs.append(f"  - {full_key}: (removed) {old[k]}")
        elif isinstance(old[k], dict) and isinstance(new[k], dict):
            diffs.extend(dict_diff(old[k], new[k], full_key))
        elif old[k] != new[k]:
            diffs.append(f"  ~ {full_key}: {old[k]} → {new[k]}")
    return diffs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # ── Load report ───────────────────────────────────────────────────────────
    if not os.path.isfile(args.report):
        print(f"[auto_improve] ERROR: report not found: {args.report}")
        sys.exit(1)
    with open(args.report, encoding="utf-8") as f:
        report = json.load(f)

    # ── Load config ───────────────────────────────────────────────────────────
    if not os.path.isfile(args.config):
        print(f"[auto_improve] ERROR: config not found: {args.config}")
        sys.exit(1)
    with open(args.config, encoding="utf-8") as f:
        old_cfg = json.load(f)

    # ── Apply rules ───────────────────────────────────────────────────────────
    new_cfg, changelog = apply_rules(old_cfg, report)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("  DustNeRF Auto-Improvement")
    print("=" * 60)

    all_suggestions = report.get("all_suggestions", [])
    if all_suggestions:
        print("\n  Analysis findings:")
        for i, s in enumerate(all_suggestions, 1):
            print(f"    {i}. {s}")

    print()
    if changelog:
        print("  Applied changes:")
        for c in changelog:
            print(c)
    else:
        print("  No hyperparameter changes needed based on current analysis.")

    print()
    full_diff = dict_diff(old_cfg, new_cfg)
    if full_diff:
        print("  Full config diff:")
        for d in full_diff:
            print(d)

    # ── Write (unless dry-run) ────────────────────────────────────────────────
    if args.dry_run:
        print("\n[auto_improve] --dry-run: config NOT written.")
    else:
        if changelog or full_diff:
            # Backup
            bak_path = args.config + ".bak"
            with open(bak_path, "w", encoding="utf-8") as f:
                json.dump(old_cfg, f, indent=2)
            print(f"\n[auto_improve] Backup → {bak_path}")

            # Write new config
            with open(args.config, "w", encoding="utf-8") as f:
                json.dump(new_cfg, f, indent=2)
            print(f"[auto_improve] Updated → {args.config}")
            print()
            print("  Next steps:")
            print("    1. Review the changes above")
            print("    2. Re-deploy to server:  bash deploy.sh all --resume")
            print("    3. Monitor:              bash deploy.sh status")
            print("    4. Download results:     bash sync_results.sh --wait")
        else:
            print("\n[auto_improve] Config unchanged.")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Apply automatic hyperparameter improvements")
    p.add_argument("--report",  default="results/analysis/analysis_report.json",
                   help="Path to analysis_report.json from src/analyze.py")
    p.add_argument("--config",  default="config/info.json",
                   help="Path to config/info.json to update")
    p.add_argument("--dry-run", action="store_true",
                   help="Print proposed changes without writing to disk")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
