"""
dataset.py — Video loading, frame extraction, background estimation,
             dust mask generation, and ray generation for DustNeRF.

Configuration split
-------------------
Training hyper-parameters (near, far, batch_rays, …) are read from the
project-level config file (``config/info.json`` by default).

Camera poses and video metadata are read from the **data-side info file**
(``<data_root>/info.json``) that is stored *alongside* the video files by
the capture pipeline.  This keeps camera-specific data out of the project
config and makes it easy to swap in a new dataset without editing any
project file.

Supported camera-entry formats in the data-side info file
----------------------------------------------------------
New format (``train_videos[]``):
    {
        "file_name": "train00.mp4",
        "frame_rate": 25.0,
        "frame_num": 113,
        "camera_angle_x": 0.389208,
        "camera_hw": [720, 1280],
        "transform_matrix": [...]
    }

Legacy format (``cameras[]``):
    {
        "id": "train00",
        "fl_x": 800.0, "fl_y": 800.0,
        "cx": 540.0,   "cy": 360.0,
        "w": 1080,     "h": 720,
        "transform_matrix": [...]
    }

Both formats are normalised to the same internal representation before
further processing.

Fallback behaviour
------------------
If ``<data_root>/info.json`` does not exist the code falls back to reading
camera entries from the project config (backwards compatible with any config
that still contains ``train_videos[]`` or ``cameras[]``).
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Config loading & format normalisation
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load the project-level config (e.g. config/info.json)."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data_info(data_root: str) -> dict:
    """
    Load the data-side info file from ``<data_root>/info.json``.

    This file is provided by the capture pipeline alongside the video files
    and contains camera poses, intrinsics, and video metadata
    (``train_videos[]`` or ``cameras[]`` entries).

    Returns an empty dict (not an error) if the file does not exist, so that
    callers can fall back to a project-config that still embeds camera entries.
    """
    info_path = os.path.join(data_root, "info.json")
    if not os.path.isfile(info_path):
        return {}
    with open(info_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[dataset] loaded data-side info from '{info_path}'")
    return data


def normalize_camera_entry(entry: dict) -> dict:
    """
    Normalise a single camera entry to the internal representation.

    Accepts either the new ``train_videos`` format (with ``camera_angle_x``
    and ``camera_hw``) or the legacy ``cameras`` format (with ``fl_x`` etc.)
    and always returns a dict with keys:
        id, fl_x, fl_y, cx, cy, w, h,
        transform_matrix, frame_rate, frame_num
    """
    if "camera_angle_x" in entry and "camera_hw" in entry:
        # ---- New format ----
        H, W = entry["camera_hw"]
        angle_x = entry["camera_angle_x"]
        fl_x = W / (2.0 * math.tan(angle_x / 2.0))
        # derive angle_y from the horizontal angle and aspect ratio
        angle_y = 2.0 * math.atan(math.tan(angle_x / 2.0) * H / W)
        fl_y = H / (2.0 * math.tan(angle_y / 2.0))
        file_name = entry["file_name"]
        cam_id = Path(file_name).stem          # "train00.mp4" → "train00"
        return {
            "id":               cam_id,
            "file_name":        file_name,
            "fl_x":             fl_x,
            "fl_y":             fl_y,
            "cx":               W / 2.0,
            "cy":               H / 2.0,
            "w":                W,
            "h":                H,
            "transform_matrix": entry["transform_matrix"],
            "frame_rate":       float(entry.get("frame_rate", 25.0)),
            "frame_num":        int(entry.get("frame_num", 1)),
        }
    else:
        # ---- Legacy format ----
        return {
            "id":               entry["id"],
            "file_name":        entry["id"] + ".mp4",
            "fl_x":             float(entry["fl_x"]),
            "fl_y":             float(entry["fl_y"]),
            "cx":               float(entry["cx"]),
            "cy":               float(entry["cy"]),
            "w":                int(entry["w"]),
            "h":                int(entry["h"]),
            "transform_matrix": entry["transform_matrix"],
            "frame_rate":       float(entry.get("frame_rate", 25.0)),
            "frame_num":        int(entry.get("frame_num",
                                              entry.get("frames_per_video", 60))),
        }


def get_camera_entries(cfg: dict, data_info: dict = None) -> List[dict]:
    """
    Return a list of normalised camera entries.

    Look-up order:
    1. ``data_info`` (the data-side ``<data_root>/info.json``) — preferred,
       because camera poses should live next to the video files, not in the
       project config.
    2. ``cfg`` (the project-level config) — fallback for backwards
       compatibility with configs that still embed ``train_videos[]`` or
       ``cameras[]``.

    Parameters
    ----------
    cfg       : project-level config dict (training hyper-parameters)
    data_info : data-side info dict loaded from ``<data_root>/info.json``;
                pass ``None`` or ``{}`` to skip.
    """
    # Prefer the data-side info file
    for source in (data_info or {}, cfg):
        raw_list = source.get("train_videos") or source.get("cameras")
        if raw_list:
            return [normalize_camera_entry(e) for e in raw_list]
    return []


# ---------------------------------------------------------------------------
# Intrinsics / extrinsics helpers
# ---------------------------------------------------------------------------

def get_intrinsics(cam: dict) -> np.ndarray:
    """Return 3×3 intrinsic matrix K from a normalised camera entry."""
    return np.array([
        [cam["fl_x"], 0.0,       cam["cx"]],
        [0.0,         cam["fl_y"], cam["cy"]],
        [0.0,         0.0,         1.0      ],
    ], dtype=np.float32)


def get_c2w(cam: dict) -> np.ndarray:
    """Return 4×4 camera-to-world matrix (NeRF convention)."""
    return np.array(cam["transform_matrix"], dtype=np.float32)


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
        stack = np.stack([f.astype(np.float32) / 255.0
                          for f in frames[:self.n_frames]], axis=0)
        if self.method == "median":
            bg = np.median(stack, axis=0)
        elif self.method == "mean":
            bg = np.mean(stack, axis=0)
        else:
            raise ValueError(f"Unknown background method: {self.method}")
        return bg.astype(np.float32)


# ---------------------------------------------------------------------------
# Video / frame-directory reader
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: str,
    max_frames: int = 120,
    frame_skip: int = 1,
    resize: Optional[Tuple[int, int]] = None,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Read frames from a video file.

    Parameters
    ----------
    video_path  : path to .mp4/.avi/etc.
    max_frames  : maximum total frames to extract
    frame_skip  : take every Nth frame (1 = every frame)
    resize      : (W, H) to resize frames, or None to keep original

    Returns
    -------
    frames     : list of H×W×3 uint8 BGR frames
    frame_idxs : list of original video frame indices for each returned frame
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames: List[np.ndarray] = []
    frame_idxs: List[int] = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_skip == 0:
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frames.append(frame)
            frame_idxs.append(idx)
        idx += 1
    cap.release()
    return frames, frame_idxs


def extract_frames_from_dir(
    frame_dir: str,
    max_frames: int = 120,
    frame_skip: int = 1,
    resize: Optional[Tuple[int, int]] = None,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load pre-extracted image frames from a directory (fallback when no video).
    Returns (frames, frame_idxs) where frame_idxs are original file indices.
    """
    paths = sorted([
        p for p in Path(frame_dir).iterdir()
        if p.suffix.lower() in extensions
    ])
    frames: List[np.ndarray] = []
    frame_idxs: List[int] = []
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
        frame_idxs.append(i)
    return frames, frame_idxs


def load_camera_frames(
    data_root: str,
    cam: dict,
    cfg_training: dict,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Attempt to load frames for a given camera from video or image directory.

    Looks for: <data_root>/<file_name>   (e.g. train00.mp4)
           OR: <data_root>/<id>/         (frame image directory)

    Returns (frames, frame_idxs).
    """
    cam_id = cam["id"]
    file_name = cam.get("file_name", cam_id + ".mp4")
    W, H = cam["w"], cam["h"]

    max_frames = cfg_training.get("frames_per_video",
                                  cam.get("frame_num", 60))
    frame_skip = cfg_training.get("frame_skip", 1)

    # Try video file named exactly as file_name, then by id with common exts
    candidates = [os.path.join(data_root, file_name)]
    if not os.path.isfile(candidates[0]):
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            candidates.append(os.path.join(data_root, cam_id + ext))

    video_path = next((c for c in candidates if os.path.isfile(c)), None)
    if video_path is not None:
        return extract_frames(video_path, max_frames=max_frames,
                              frame_skip=frame_skip, resize=(W, H))

    frame_dir = os.path.join(data_root, cam_id)
    if os.path.isdir(frame_dir):
        return extract_frames_from_dir(frame_dir, max_frames=max_frames,
                                       frame_skip=frame_skip, resize=(W, H))

    raise FileNotFoundError(
        f"No video or frame directory found for camera '{cam_id}' "
        f"under '{data_root}'. "
        f"Expected '{data_root}/{file_name}' or '{data_root}/{cam_id}/'."
    )


# ---------------------------------------------------------------------------
# Ray generation
# ---------------------------------------------------------------------------

def get_rays(
    H: int, W: int, K: np.ndarray, c2w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate per-pixel rays in world space.

    Returns
    -------
    rays_o : (H*W, 3) ray origins
    rays_d : (H*W, 3) ray directions (unit vectors)
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32), indexing="xy")
    dirs = np.stack([
        (i - K[0, 2]) / K[0, 0],
        -(j - K[1, 2]) / K[1, 1],
        -np.ones_like(i),
    ], axis=-1)                                            # H×W×3
    rays_d = (c2w[:3, :3] @ dirs[..., np.newaxis]).squeeze(-1)  # H×W×3
    rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape).copy()   # H×W×3
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
    weight = diff.mean(axis=-1)
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
    PyTorch dataset providing (ray_origin, ray_direction, rgb, dust_weight,
    frame_idx) tuples for every pixel of every frame of every camera.

    ``frame_idx`` is the **original video frame number** (0-based), shared
    across all cameras at the same time step.  It is used as a time index
    for the temporal dust-probability head in DustNeRF.

    Attributes
    ----------
    n_frames : int
        Total number of distinct temporal frames across all cameras
        (= ``frame_num`` from config, or the maximum observed).
    frame_times : np.ndarray
        (n_frames,) array of timestamps in seconds, computed from
        ``frame_rate``.  All cameras are assumed synchronised.
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

        # Load camera poses and video metadata from the data-side info file.
        # Falls back to the project config if the file does not exist.
        data_info = load_data_info(data_root)

        bg_estimator = BackgroundEstimator(n_frames=bg_frames, method=bg_method)

        all_rays_o:    List[np.ndarray] = []
        all_rays_d:    List[np.ndarray] = []
        all_rgb:       List[np.ndarray] = []
        all_weight:    List[np.ndarray] = []
        all_frame_idx: List[np.ndarray] = []

        max_frame_idx = 0
        frame_rate = 25.0   # default; overwritten by first camera entry

        camera_entries = get_camera_entries(self.cfg, data_info)
        print(f"[DustDataset] Loading data from '{data_root}' …")

        for cam in camera_entries:
            cam_id = cam["id"]
            H, W   = cam["h"], cam["w"]
            K      = get_intrinsics(cam)
            c2w    = get_c2w(cam)
            frame_rate = cam["frame_rate"]

            # ----- load frames -----
            try:
                frames, orig_frame_idxs = load_camera_frames(
                    data_root, cam, cfg_train)
            except FileNotFoundError as e:
                print(f"  [WARN] {e}  — skipping camera {cam_id}")
                continue

            if len(frames) == 0:
                print(f"  [WARN] No frames found for {cam_id} — skipping")
                continue

            print(f"  [{cam_id}] loaded {len(frames)} frames"
                  f"  (H={H}, W={W},"
                  f" frames {orig_frame_idxs[0]}–{orig_frame_idxs[-1]})")

            # ----- background estimation -----
            bg_bgr = bg_estimator.estimate(frames)

            # ----- rays (same for every frame of this camera) -----
            rays_o, rays_d = get_rays(H, W, K, c2w)

            # ----- per-frame rays -----
            for frame, orig_idx in zip(frames, orig_frame_idxs):
                frame_resized = cv2.resize(frame, (W, H),
                                           interpolation=cv2.INTER_AREA)
                rgb = frame_resized[:, :, ::-1].astype(np.float32) / 255.0
                rgb_flat    = rgb.reshape(-1, 3)
                weight      = compute_dust_weight(frame_resized, bg_bgr,
                                                  threshold=dust_threshold)
                weight_flat = weight.reshape(-1)

                n_px = H * W
                fidx_flat = np.full(n_px, orig_idx, dtype=np.int64)

                all_rays_o.append(rays_o)
                all_rays_d.append(rays_d)
                all_rgb.append(rgb_flat)
                all_weight.append(weight_flat)
                all_frame_idx.append(fidx_flat)

                if orig_idx > max_frame_idx:
                    max_frame_idx = orig_idx

        if not all_rays_o:
            raise RuntimeError(
                "No data loaded.  Check that data_root contains video files or "
                "frame directories named 'train00', 'train01', … "
                "(or matching file_name fields in config)."
            )

        self.rays_o    = torch.from_numpy(np.concatenate(all_rays_o,    axis=0))
        self.rays_d    = torch.from_numpy(np.concatenate(all_rays_d,    axis=0))
        self.rgb       = torch.from_numpy(np.concatenate(all_rgb,       axis=0))
        self.weight    = torch.from_numpy(np.concatenate(all_weight,    axis=0))
        self.frame_idx = torch.from_numpy(np.concatenate(all_frame_idx, axis=0))

        # Total number of distinct time steps.
        # Use the maximum frame_num declared in any camera entry (all cameras
        # are synchronised, so they should all have the same value).  Fall back
        # to the highest observed frame index + 1 if no camera declares frame_num.
        declared_frame_num = max(
            (cam.get("frame_num", 0) for cam in camera_entries),
            default=0,
        )
        self.n_frames = int(declared_frame_num) if declared_frame_num > 0 \
                        else (max_frame_idx + 1)
        self.frame_times = np.arange(self.n_frames, dtype=np.float32) / frame_rate

        total = len(self.rays_o)
        print(f"[DustDataset] Total rays: {total:,}  |  n_frames={self.n_frames}")

    def __len__(self) -> int:
        return len(self.rays_o)

    def __getitem__(self, idx):
        return {
            "rays_o":    self.rays_o[idx],
            "rays_d":    self.rays_d[idx],
            "rgb":       self.rgb[idx],
            "weight":    self.weight[idx],
            "frame_idx": self.frame_idx[idx],
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
