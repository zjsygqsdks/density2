"""
dataset.py — Video loading, frame extraction, background estimation,
             dust mask generation, and ray generation for DustNeRF.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load config/info.json."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_intrinsics(cam_cfg: dict) -> np.ndarray:
    """Return 3×3 intrinsic matrix K from a camera config entry."""
    K = np.array([
        [cam_cfg["fl_x"],           0.0, cam_cfg["cx"]],
        [          0.0, cam_cfg["fl_y"], cam_cfg["cy"]],
        [          0.0,           0.0,           1.0],
    ], dtype=np.float32)
    return K


def get_c2w(cam_cfg: dict) -> np.ndarray:
    """Return 4×4 camera-to-world matrix (NeRF convention)."""
    return np.array(cam_cfg["transform_matrix"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Background estimation
# ---------------------------------------------------------------------------

class BackgroundEstimator:
    """Estimate per-pixel background from the first N static frames of a video."""

    def __init__(self, n_frames: int = 30, method: str = "median"):
        self.n_frames = n_frames
        self.method = method

    def estimate(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        frames : list of H×W×3 uint8 arrays
        Returns : H×W×3 float32 background in [0,1]
        """
        stack = np.stack([f.astype(np.float32) / 255.0 for f in frames[:self.n_frames]], axis=0)
        if self.method == "median":
            bg = np.median(stack, axis=0)
        elif self.method == "mean":
            bg = np.mean(stack, axis=0)
        else:
            raise ValueError(f"Unknown background method: {self.method}")
        return bg.astype(np.float32)


# ---------------------------------------------------------------------------
# Video reader
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: str,
    max_frames: int = 120,
    frame_skip: int = 5,
    resize: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """
    Read frames from a video file.

    Parameters
    ----------
    video_path : path to .mp4/.avi/etc.
    max_frames : maximum total frames to extract
    frame_skip : take every Nth frame
    resize     : (W, H) to resize frames, or None to keep original

    Returns
    -------
    list of H×W×3 uint8 BGR frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_skip == 0:
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def extract_frames_from_dir(
    frame_dir: str,
    max_frames: int = 120,
    frame_skip: int = 5,
    resize: Optional[Tuple[int, int]] = None,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> List[np.ndarray]:
    """
    Load pre-extracted image frames from a directory (fallback when no video).
    """
    paths = sorted([
        p for p in Path(frame_dir).iterdir()
        if p.suffix.lower() in extensions
    ])
    frames = []
    for i, p in enumerate(paths):
        if len(frames) >= max_frames:
            break
        if i % frame_skip != 0:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        if resize is not None:
            img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
        frames.append(img)
    return frames


def load_camera_frames(
    data_root: str,
    cam_id: str,
    cfg_training: dict,
    cam_wh: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """
    Attempt to load frames for a given camera from video or image directory.
    Looks for: <data_root>/<cam_id>.mp4  OR  <data_root>/<cam_id>/
    """
    max_frames = cfg_training.get("frames_per_video", 60)
    frame_skip = cfg_training.get("frame_skip", 5)

    video_exts = (".mp4", ".avi", ".mov", ".mkv")
    video_path = None
    for ext in video_exts:
        candidate = os.path.join(data_root, cam_id + ext)
        if os.path.isfile(candidate):
            video_path = candidate
            break

    if video_path is not None:
        return extract_frames(video_path, max_frames=max_frames,
                              frame_skip=frame_skip, resize=cam_wh)

    frame_dir = os.path.join(data_root, cam_id)
    if os.path.isdir(frame_dir):
        return extract_frames_from_dir(frame_dir, max_frames=max_frames,
                                       frame_skip=frame_skip, resize=cam_wh)

    raise FileNotFoundError(
        f"No video or frame directory found for camera '{cam_id}' "
        f"under '{data_root}'. "
        f"Expected '{data_root}/{cam_id}.mp4' or '{data_root}/{cam_id}/'."
    )


# ---------------------------------------------------------------------------
# Ray generation
# ---------------------------------------------------------------------------

def get_rays(H: int, W: int, K: np.ndarray, c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate per-pixel rays in world space.

    Returns
    -------
    rays_o : (H*W, 3) ray origins
    rays_d : (H*W, 3) ray directions (unit vectors)
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing="xy")
    # un-project to camera space
    dirs = np.stack([
        (i - K[0, 2]) / K[0, 0],
        -(j - K[1, 2]) / K[1, 1],
        -np.ones_like(i),
    ], axis=-1)  # H×W×3
    # rotate to world space
    rays_d = (c2w[:3, :3] @ dirs[..., np.newaxis]).squeeze(-1)  # H×W×3
    # normalise
    rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape).copy()  # H×W×3
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


# ---------------------------------------------------------------------------
# Dust weight (per-pixel foreground importance)
# ---------------------------------------------------------------------------

def compute_dust_weight(
    frame_bgr: np.ndarray,
    background_bgr: np.ndarray,
    threshold: float = 0.05,
) -> np.ndarray:
    """
    Compute per-pixel dust weight as residual between frame and background.

    Parameters
    ----------
    frame_bgr      : H×W×3 uint8
    background_bgr : H×W×3 float32 in [0,1]
    threshold      : ignore residuals below this value

    Returns
    -------
    weight : H×W float32 in [0,1]
    """
    frame_f = frame_bgr.astype(np.float32) / 255.0
    diff = np.abs(frame_f - background_bgr)
    weight = diff.mean(axis=-1)           # average over RGB channels
    weight = np.clip(weight - threshold, 0.0, None)
    max_w = weight.max()
    if max_w > 0:
        weight = weight / max_w
    return weight.astype(np.float32)


# ---------------------------------------------------------------------------
# Main Dataset class
# ---------------------------------------------------------------------------

class DustDataset(Dataset):
    """
    PyTorch dataset that provides (ray_origin, ray_direction, rgb, dust_weight) tuples.

    Each sample is one pixel/ray from one frame of one camera.
    """

    def __init__(
        self,
        config_path: str,
        data_root: str,
        bg_frames: int = 30,
        bg_method: str = "median",
        dust_threshold: float = 0.05,
        device: str = "cpu",
    ):
        super().__init__()
        self.cfg = load_config(config_path)
        self.data_root = data_root
        self.device = device
        cfg_train = self.cfg["training"]

        bg_estimator = BackgroundEstimator(n_frames=bg_frames, method=bg_method)

        all_rays_o: List[np.ndarray] = []
        all_rays_d: List[np.ndarray] = []
        all_rgb: List[np.ndarray] = []
        all_weight: List[np.ndarray] = []

        print(f"[DustDataset] Loading data from '{data_root}' …")
        for cam_cfg in self.cfg["cameras"]:
            cam_id = cam_cfg["id"]
            H, W = cam_cfg["h"], cam_cfg["w"]
            K = get_intrinsics(cam_cfg)
            c2w = get_c2w(cam_cfg)

            # ----- load frames -----
            try:
                frames = load_camera_frames(data_root, cam_id, cfg_train, cam_wh=(W, H))
            except FileNotFoundError as e:
                print(f"  [WARN] {e}  — skipping camera {cam_id}")
                continue

            if len(frames) == 0:
                print(f"  [WARN] No frames found for {cam_id} — skipping")
                continue

            print(f"  [{cam_id}] loaded {len(frames)} frames  (H={H}, W={W})")

            # ----- background estimation -----
            bg_bgr = bg_estimator.estimate(frames)

            # ----- rays (same for every frame of this camera) -----
            rays_o, rays_d = get_rays(H, W, K, c2w)

            # ----- per-frame rays -----
            for frame in frames:
                # resize frame to configured resolution
                frame_resized = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                # RGB in [0,1]
                rgb = frame_resized[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB
                rgb_flat = rgb.reshape(-1, 3)
                # dust weight
                weight = compute_dust_weight(frame_resized, bg_bgr, threshold=dust_threshold)
                weight_flat = weight.reshape(-1)

                all_rays_o.append(rays_o)
                all_rays_d.append(rays_d)
                all_rgb.append(rgb_flat)
                all_weight.append(weight_flat)

        if not all_rays_o:
            raise RuntimeError(
                "No data loaded. Check that data_root contains video files or frame directories "
                "named 'train00', 'train01', …, 'train05'."
            )

        self.rays_o = torch.from_numpy(np.concatenate(all_rays_o, axis=0))   # N×3
        self.rays_d = torch.from_numpy(np.concatenate(all_rays_d, axis=0))   # N×3
        self.rgb    = torch.from_numpy(np.concatenate(all_rgb,    axis=0))   # N×3
        self.weight = torch.from_numpy(np.concatenate(all_weight, axis=0))   # N

        total = len(self.rays_o)
        print(f"[DustDataset] Total rays: {total:,}")

    def __len__(self) -> int:
        return len(self.rays_o)

    def __getitem__(self, idx):
        return {
            "rays_o": self.rays_o[idx],
            "rays_d": self.rays_d[idx],
            "rgb":    self.rgb[idx],
            "weight": self.weight[idx],
        }

    # ------------------------------------------------------------------
    # Convenience: build a scene bounding box from all ray origins
    # ------------------------------------------------------------------
    def scene_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scene_min, scene_max) as (3,) arrays."""
        origins = self.rays_o.numpy()
        scene_min = origins.min(axis=0) - 1.0
        scene_max = origins.max(axis=0) + 1.0
        return scene_min, scene_max
