"""
train.py — Training loop for DustNeRF.

Usage (headless server):
    python -m src.train --config config/info.json --data data/ --out outputs/

Features
--------
* Hierarchical NeRF training (coarse + fine)
* Dust-weighted photometric loss
* Periodic checkpointing
* TensorBoard logging (no display required)
* Automatic scene-bounds detection from camera positions
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .dataset import DustDataset, load_config
from .model import DustNeRF
from .renderer import render_rays


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def photometric_loss(
    pred_rgb: torch.Tensor,   # (B, 3)
    gt_rgb: torch.Tensor,     # (B, 3)
    weight: torch.Tensor,     # (B,)  dust importance weight
    dust_weight_alpha: float = 5.0,
) -> torch.Tensor:
    """
    Weighted MSE loss.  Pixels with high dust weight contribute more.
    """
    w = 1.0 + dust_weight_alpha * weight           # (B,) ≥ 1
    mse = ((pred_rgb - gt_rgb) ** 2).mean(dim=-1)  # (B,)
    return (w * mse).mean()


def dust_regularisation(dust_map: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Encourage dust_map to be high where dust_weight is high, low otherwise.
    """
    return F.mse_loss(dust_map, weight.clamp(0.0, 1.0))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, cfg_train: dict):
    """Exponential LR decay."""
    decay_steps = cfg_train.get("lr_decay_steps", 250000)
    decay_factor = cfg_train.get("lr_decay_factor", 0.1)
    gamma = decay_factor ** (1.0 / decay_steps)
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


def save_checkpoint(
    out_dir: str,
    step: int,
    coarse: DustNeRF,
    fine: DustNeRF,
    optimizer,
    scheduler,
):
    ckpt = {
        "step": step,
        "coarse": coarse.state_dict(),
        "fine": fine.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    path = os.path.join(out_dir, f"ckpt_{step:07d}.pt")
    torch.save(ckpt, path)
    # also save as latest
    torch.save(ckpt, os.path.join(out_dir, "ckpt_latest.pt"))
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(
    path: str,
    coarse: DustNeRF,
    fine: DustNeRF,
    optimizer,
    scheduler,
    device: torch.device,
) -> int:
    ckpt = torch.load(path, map_location=device)
    coarse.load_state_dict(ckpt["coarse"])
    fine.load_state_dict(ckpt["fine"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    step = ckpt["step"]
    print(f"  [ckpt] resumed from step {step} ← {path}")
    return step


# ---------------------------------------------------------------------------
# JSONL metrics logger (for offline analysis via src/analyze.py)
# ---------------------------------------------------------------------------

class JsonlLogger:
    """Append one JSON record per log step to <out_dir>/train_metrics.jsonl."""

    def __init__(self, path: str):
        self.path = path

    def log(self, record: dict):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    # ---- Config ----
    cfg = load_config(args.config)
    cfg_train = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # ---- Dataset ----
    dataset = DustDataset(
        config_path=args.config,
        data_root=args.data,
        bg_frames=cfg_train.get("background_frames", 30),
        bg_method="median",
        dust_threshold=cfg_train.get("dust_threshold", 0.05),
        device="cpu",
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg_train["batch_rays"],
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # ---- Models ----
    coarse = DustNeRF(pos_freqs=10, dir_freqs=4, net_depth=8, net_width=256).to(device)
    fine   = DustNeRF(pos_freqs=10, dir_freqs=4, net_depth=8, net_width=256).to(device)

    total_params = sum(p.numel() for p in list(coarse.parameters()) + list(fine.parameters()))
    print(f"[train] model parameters: {total_params:,}")

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(
        list(coarse.parameters()) + list(fine.parameters()),
        lr=cfg_train.get("learning_rate", 5e-4),
    )
    scheduler = build_scheduler(optimizer, cfg_train)

    # ---- Resume ----
    start_step = 0
    latest_ckpt = ckpt_dir / "ckpt_latest.pt"
    if args.resume and latest_ckpt.exists():
        start_step = load_checkpoint(str(latest_ckpt), coarse, fine, optimizer, scheduler, device)

    # ---- TensorBoard ----
    writer = SummaryWriter(log_dir=str(out_dir / "logs"))

    # ---- JSONL metrics logger (offline-analysis friendly) ----
    jsonl_logger = JsonlLogger(str(out_dir / "train_metrics.jsonl"))

    # ---- Render settings ----
    near = cfg_train["near"]
    far  = cfg_train["far"]
    n_coarse = cfg_train["n_samples_coarse"]
    n_fine   = cfg_train["n_samples_fine"]
    max_steps = cfg_train["max_steps"]
    save_every = cfg_train["save_every"]
    log_every  = cfg_train["log_every"]
    dust_weight_alpha = cfg_train.get("dust_weight_alpha", 5.0)

    # ---- Training loop ----
    step = start_step
    loader_iter = iter(loader)
    t0 = time.time()

    print(f"[train] starting at step {step}, max_steps={max_steps}")

    while step < max_steps:
        # cycle through dataset
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        rays_o  = batch["rays_o"].to(device)   # (B, 3)
        rays_d  = batch["rays_d"].to(device)   # (B, 3)
        gt_rgb  = batch["rgb"].to(device)      # (B, 3)
        weight  = batch["weight"].to(device)   # (B,)

        # ---- forward ----
        out = render_rays(
            coarse, fine,
            rays_o, rays_d,
            near=near, far=far,
            n_coarse=n_coarse, n_fine=n_fine,
            perturb=True, white_bkgd=args.white_bkgd,
        )

        loss_c = photometric_loss(out["coarse/rgb"], gt_rgb, weight, dust_weight_alpha)
        loss_f = photometric_loss(out["fine/rgb"],   gt_rgb, weight, dust_weight_alpha)
        loss_dust = dust_regularisation(out["fine/dust"], weight)
        loss = loss_c + loss_f + 0.1 * loss_dust

        # ---- backward ----
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(coarse.parameters()) + list(fine.parameters()), 1.0
        )
        optimizer.step()
        scheduler.step()

        step += 1

        # ---- logging ----
        if step % log_every == 0:
            psnr = -10.0 * torch.log10(loss_f.detach() + 1e-8).item()
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            print(
                f"  step {step:7d}/{max_steps}  "
                f"loss={loss.item():.4f}  psnr={psnr:.2f} dB  "
                f"lr={lr:.2e}  {elapsed:.1f}s"
            )
            writer.add_scalar("loss/total",   loss.item(),   step)
            writer.add_scalar("loss/coarse",  loss_c.item(), step)
            writer.add_scalar("loss/fine",    loss_f.item(), step)
            writer.add_scalar("loss/dust_reg",loss_dust.item(), step)
            writer.add_scalar("psnr/fine",    psnr,          step)
            writer.add_scalar("lr",           lr,            step)

            jsonl_logger.log({
                "step":      step,
                "loss":      round(loss.item(),      6),
                "loss_c":    round(loss_c.item(),    6),
                "loss_f":    round(loss_f.item(),    6),
                "loss_dust": round(loss_dust.item(), 6),
                "psnr":      round(psnr,             4),
                "lr":        lr,
                "elapsed_s": round(elapsed,          1),
            })

        # ---- checkpoint ----
        if step % save_every == 0 or step == max_steps:
            save_checkpoint(str(ckpt_dir), step, coarse, fine, optimizer, scheduler)

    writer.close()
    print(f"[train] done. Checkpoints in {ckpt_dir}")
    return coarse, fine


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train DustNeRF")
    p.add_argument("--config", default="config/info.json", help="Path to config/info.json")
    p.add_argument("--data",   default="data",             help="Root dir of video/frame data")
    p.add_argument("--out",    default="outputs",          help="Output directory")
    p.add_argument("--resume", action="store_true",        help="Resume from latest checkpoint")
    p.add_argument("--white-bkgd", action="store_true",    help="Use white background")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
