# density2 — Spatial Dust Distribution Reconstruction

A machine-learning pipeline that reconstructs the **3-D spatial distribution of airborne dust** from six synchronized camera videos using Neural Radiance Fields (NeRF).

---

## Overview

```
data/
  train00.mp4   ← camera at   0°
  train01.mp4   ← camera at  60°
  train02.mp4   ← camera at 120°
  train03.mp4   ← camera at 180°
  train04.mp4   ← camera at 240°
  train05.mp4   ← camera at 300°
config/
  info.json     ← NeRF-format camera poses + intrinsics + training config
outputs/        ← checkpoints, logs, exports (auto-created)
```

Six cameras are placed in a horizontal ring with **60° spacing**.  
Each camera has **unique intrinsics** (focal length, principal point).  
Camera poses (`transform_matrix`) follow the **NeRF convention** (camera-to-world, right-hand coordinate system, +Y up, −Z forward).

---

## How It Works

### Stage 1 — Background subtraction
Each video's first N frames are averaged (median) to estimate a static background image. Per-pixel **dust weights** are computed as the L1 colour residual between each frame and the background, which helps the network focus on transient dust particles rather than the complex static scene.

### Stage 2 — Ray generation
For every pixel in every frame, a ray is cast from the camera origin through the pixel in world space using the NeRF-style volume-rendering framework.

### Stage 3 — DustNeRF training
A two-level (coarse + fine) **DustNeRF MLP** is trained:

```
Input:  (x, y, z)  +  viewing direction d
Output: σ (density)  +  RGB colour  +  dust probability
```

- **Positional encoding** (Fourier features, 10 frequencies for position, 4 for direction)
- **Dust-weighted photometric loss**: pixels with high dust weight contribute 5× more to the loss
- **Dust regularisation loss**: `dust_prob` is penalised to match the background-subtracted dust mask
- **Hierarchical sampling**: coarse network guides importance sampling for the fine network

### Stage 4 — Export
After training, the fine model is queried on a **128³ regular grid** and results are saved as:

| File | Description |
|------|-------------|
| `density_grid.npz` | Compressed NumPy archive: `density`, `dust_prob`, `grid_min`, `grid_max` |
| `density_grid.vtk` | VTK legacy ASCII — open in **ParaView** or **VisIt** |
| `cameras.json` | Camera poses and intrinsics |
| `dust_cloud.html` | Interactive **Plotly 3-D scatter** (open in any browser) |
| `dust_density_slices.png` | Matplotlib mid-plane density slices |
| `dust_export.tar.gz` | Single archive of all of the above for easy download |

---

## Quick Start

### 1. Install dependencies (server)

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Place your videos (or extracted frame directories) under `data/`:

```
data/
  train00.mp4      # or train00/ containing frame_0000.png, ...
  train01.mp4
  ...
  train05.mp4
```

Edit `config/info.json` to match your actual camera intrinsics and poses if different from the defaults.

### 3. Train (headless server)

```bash
# Full pipeline: train + export
bash run.sh all

# Train only
bash run.sh train

# Resume interrupted training
bash run.sh train --resume

# Export only (after training)
bash run.sh export
```

Custom paths:
```bash
CONFIG=config/info.json DATA=data OUT=outputs bash run.sh all
```

### 4. Download and visualise (local machine)

```bash
# Download archive from server
scp user@server:outputs/dust_export.tar.gz .

# Extract
tar -xzf dust_export.tar.gz

# Install local viz dependencies
pip install numpy matplotlib plotly scipy

# Launch visualisation
python visualize_local.py --export export/

# Save to files instead of opening windows
python visualize_local.py --export export/ \
    --save-html dust_cloud.html \
    --save-png  dust_slices.png
```

---

## Remote Server Workflow (Deploy → Train → Download → Analyze → Improve)

This is the recommended workflow when you have a dedicated training server.

### 0. Configure server access

```bash
cp server.env.example server.env
# Edit server.env with your server details:
nano server.env
```

```ini
REMOTE_HOST=192.168.1.100   # server IP or hostname
REMOTE_PORT=22              # SSH port
REMOTE_USER=ubuntu          # SSH username
REMOTE_DIR=/home/ubuntu/density2
```

`server.env` is in `.gitignore` — your credentials are never committed.

---

### 1. Deploy and start training

```bash
# Upload project + data, install deps, start training in background
bash deploy.sh all

# Or train only (without export):
bash deploy.sh train
```

Monitor the remote training log:
```bash
bash deploy.sh status
ssh -p 22 ubuntu@192.168.1.100 tail -f /home/ubuntu/density2/train.log
```

---

### 2. Download results and analyze

```bash
# Wait for training to finish, then download + analyze + auto-improve
bash sync_results.sh --wait

# Or download immediately (if training already done):
bash sync_results.sh
```

`sync_results.sh` automatically:
1. Polls the server until training finishes (with `--wait`)
2. Triggers the export step on the server
3. Downloads `dust_export.tar.gz` + `train_metrics.jsonl` + `train.log`
4. Extracts the archive to `results/export/`
5. Runs `src/analyze.py` → `results/analysis/`
6. Runs `auto_improve.py` → patches `config/info.json`

---

### 3. Review analysis results

```
results/
  export/
    density_grid.npz           ← 3-D density + dust_prob volume
    dust_cloud.html            ← interactive 3-D viewer
    dust_density_slices.png    ← mid-plane slices
  analysis/
    analysis_report.json       ← findings + suggestions (machine-readable)
    training_curves.png        ← loss / PSNR curves
    density_histogram.png      ← density distribution
    dust_coverage_map.png      ← top-view projection
```

Open `results/export/dust_cloud.html` in any browser to see the 3-D dust cloud.

Run the full interactive viewer:
```bash
python visualize_local.py --export results/export/
```

---

### 4. Auto-improve and redeploy

`sync_results.sh` calls `auto_improve.py` automatically, but you can also run it manually:

```bash
# Dry-run: see what would change
python auto_improve.py --report results/analysis/analysis_report.json --dry-run

# Apply changes to config/info.json
python auto_improve.py --report results/analysis/analysis_report.json
```

The script applies these rules automatically:

| Finding | Applied fix |
|---------|-------------|
| Training diverged | Halve `learning_rate` |
| PSNR < 20 dB | Increase `max_steps` by 50% |
| Training plateau | More steps + slower `lr_decay_factor` |
| Dust coverage < 0.01% | Lower `dust_threshold`, fewer `background_frames` |
| Dust coverage > 30% | Raise `dust_threshold`, more `background_frames` |
| Volume near-empty | Double coarse + fine sample counts |
| `dust_prob` mean < 0.05 | Double `dust_weight_alpha` |
| PSNR ≥ 25 dB | Increase export `grid_resolution` to 256 |

After reviewing the changes, redeploy with resume:

```bash
bash deploy.sh train
# After training:
bash sync_results.sh --wait
```

---

### Full cycle at a glance

```
local                          server
─────                          ──────
bash deploy.sh all  ─rsync──►  installs deps
                    ─start──►  nohup run.sh train &
bash deploy.sh status ◄─────── tail train.log
bash sync_results.sh --wait
                    ◄─scp────  dust_export.tar.gz
                    ◄─scp────  train_metrics.jsonl
src/analyze.py     (local)     → results/analysis/
auto_improve.py    (local)     → updates config/info.json
bash deploy.sh train ─rsync──► updated config
                    ─start──►  nohup run.sh train --resume &
  ... (repeat until quality is satisfactory)
```

---


### Training directly with Python

```bash
python -m src.train \
    --config config/info.json \
    --data   data/ \
    --out    outputs/ \
    --resume \
    --white-bkgd
```

### Export with custom checkpoint

```bash
python -m src.export \
    --config config/info.json \
    --ckpt   outputs/checkpoints/ckpt_0050000.pt \
    --out    outputs/
```

### TensorBoard monitoring (headless-friendly)

```bash
# On server — start TensorBoard without GUI
tensorboard --logdir outputs/logs --host 0.0.0.0 --port 6006

# On local machine — SSH port forward
ssh -L 6006:localhost:6006 user@server

# Then open in browser: http://localhost:6006
```

---

## Configuration Reference (`config/info.json`)

### `cameras[]`

| Field | Description |
|-------|-------------|
| `id` | Camera name (`train00`–`train05`) |
| `fl_x`, `fl_y` | Focal lengths in pixels |
| `cx`, `cy` | Principal point in pixels |
| `w`, `h` | Image width/height in pixels |
| `transform_matrix` | 4×4 camera-to-world matrix (NeRF convention) |

### `training`

| Field | Default | Description |
|-------|---------|-------------|
| `near` / `far` | 0.1 / 10.0 | Ray near/far bounds (metres) |
| `n_samples_coarse` | 64 | Coarse samples per ray |
| `n_samples_fine` | 128 | Fine samples per ray |
| `batch_rays` | 1024 | Training batch size (rays) |
| `learning_rate` | 5e-4 | Initial Adam learning rate |
| `max_steps` | 200 000 | Total training steps |
| `frames_per_video` | 60 | Max frames extracted per video |
| `frame_skip` | 5 | Extract every Nth frame |
| `background_frames` | 30 | Frames used for background estimation |
| `dust_threshold` | 0.05 | Min residual to count as dust |
| `dust_weight_alpha` | 5.0 | Loss multiplier for dusty pixels (auto-tuned) |

### `export`

| Field | Default | Description |
|-------|---------|-------------|
| `grid_resolution` | 128 | Voxel grid resolution per axis |
| `density_threshold` | 0.01 | Threshold for HTML scatter display |

---

## Project Structure

```
density2/
├── config/
│   └── info.json          # NeRF camera config + training hyperparameters
├── src/
│   ├── __init__.py
│   ├── dataset.py         # Video loading, background subtraction, ray generation
│   ├── model.py           # DustNeRF MLP (pos. encoding → σ + RGB + dust_prob)
│   ├── renderer.py        # Differentiable volume rendering (coarse + fine)
│   ├── train.py           # Training loop + TensorBoard + JSONL logging + checkpointing
│   ├── export.py          # 3-D grid evaluation + VTK/NPZ/HTML/PNG export
│   └── analyze.py         # Offline analysis: training curves + density grid statistics
├── auto_improve.py        # Read analysis report → patch config/info.json automatically
├── deploy.sh              # Upload to remote server and start training
├── sync_results.sh        # Download results + run analysis + run auto-improvement
├── server.env.example     # Template for server credentials (copy to server.env)
├── visualize_local.py     # Local visualisation (Plotly 3-D + Matplotlib slices)
├── run.sh                 # Headless server entry point
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Notes on Complex Backgrounds

Because real-world environments are non-uniform, the pipeline uses:

1. **Temporal median background subtraction** — robust to transient events in the first `background_frames` frames.
2. **Dust-weighted loss** — network learns to model foreground dust more precisely.
3. **`dust_prob` output head** — a separate probability branch isolates dust voxels from structural scene density, enabling clean thresholding during export.

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (GPU strongly recommended)
- OpenCV (headless), NumPy, SciPy, tqdm, imageio, matplotlib, plotly, tensorboard

GPU memory: ~4 GB for batch_rays=1024; reduce batch size if needed.
