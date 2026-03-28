# DustNeRF — 程序原理与完整运行流程

> **语言 / Language**：本文档以中文为主，关键技术术语附英文对照。  
> **对应版本**：`density2` 仓库当前主分支。

---

## 目录

1. [项目背景与目标](#1-项目背景与目标)
2. [核心数学原理](#2-核心数学原理)
   - 2.1 [体渲染积分（Volume Rendering Integral）](#21-体渲染积分)
   - 2.2 [位置编码（Positional Encoding）](#22-位置编码)
   - 2.3 [分层重要性采样（Hierarchical Importance Sampling）](#23-分层重要性采样)
   - 2.4 [沙尘加权损失（Dust-Weighted Loss）](#24-沙尘加权损失)
3. [代码模块详解](#3-代码模块详解)
   - 3.1 [dataset.py — 数据加载与光线生成](#31-datasetpy)
   - 3.2 [model.py — DustNeRF 神经网络](#32-modelpy)
   - 3.3 [renderer.py — 可微体渲染](#33-rendererpy)
   - 3.4 [train.py — 训练循环](#34-trainpy)
   - 3.5 [export.py — 密度网格导出](#35-exportpy)
   - 3.6 [analyze.py — 训练结果分析](#36-analyzepy)
   - 3.7 [auto_improve.py — 自动超参改进](#37-auto_improvepy)
   - 3.8 [visualize_local.py — 本地可视化](#38-visualize_localpy)
4. [完整运行流程](#4-完整运行流程)
   - 4.1 [数据准备](#41-数据准备)
   - 4.2 [本地单机运行](#42-本地单机运行)
   - 4.3 [远程服务器运行（推荐）](#43-远程服务器运行推荐)
   - 4.4 [下载与分析](#44-下载与分析)
   - 4.5 [自动改进后重训](#45-自动改进后重训)
5. [配置文件详解（config/info.json）](#5-配置文件详解)
6. [输出文件说明](#6-输出文件说明)
7. [数据流程总览图](#7-数据流程总览图)
8. [常见问题与调参指南](#8-常见问题与调参指南)

---

## 1. 项目背景与目标

**DustNeRF**（Dust Neural Radiance Field）是一套利用 **神经辐射场（NeRF）** 技术，从多摄像机视频中重建**空气中三维沙尘/颗粒物分布**的机器学习流水线。

### 实际应用场景

| 场景 | 说明 |
|------|------|
| 工业粉尘监测 | 厂房或矿山中实时重建粉尘三维分布 |
| 大气颗粒物研究 | 室外/风洞实验中颗粒物空间分析 |
| 消防/烟雾仿真 | 可视化烟尘扩散路径 |

### 系统输入输出

```
输入：六路同步摄像机视频（60° 等间距水平环形排列）
        ↓
输出：三维密度体素网格（density + dust_prob）
       + 交互式三维可视化 HTML
       + 密度切片图
       + VTK 格式（可用 ParaView 查看）
```

---

## 2. 核心数学原理

### 2.1 体渲染积分

NeRF 的核心公式是**经典体渲染方程**（来自计算机图形学）。
给定一条从摄像机出发、方向为 **d** 的光线，其起点为 **o**，则在该光线上距离 *t* 处的三维坐标为：

```
r(t) = o + t·d
```

沿光线的颜色（C）由以下积分决定：

```
C(r) = ∫[t_near → t_far]  T(t) · σ(r(t)) · c(r(t), d)  dt

其中透射率：
T(t) = exp( -∫[t_near → t]  σ(r(s)) ds )
```

- **σ（sigma）**：体素密度（不透明度）——越大表示该点越"实"
- **c**：该点发出的颜色（RGB）
- **T(t)**：从起点到 *t* 的透射率（光线还未被遮挡的概率）

**数值离散化**：将光线离散为 S 个采样点 {t₁, t₂, …, tₛ}，相邻采样间距为 δᵢ：

```
αᵢ = 1 − exp(−σᵢ · δᵢ)          （单段不透明度）
Tᵢ = ∏[j<i] (1 − αⱼ)             （前缀透射率）
wᵢ = Tᵢ · αᵢ                      （体渲染权重，∑wᵢ ≤ 1）

C  = ∑ᵢ wᵢ · cᵢ                   （渲染颜色）
D  = ∑ᵢ wᵢ · tᵢ                   （期望深度）
A  = ∑ᵢ wᵢ                        （累积不透明度）
```

**沙尘密度积分**（DustNeRF 新增）：

```
dust_map = ∑ᵢ wᵢ · dust_probᵢ
```

这使每条光线同时得到一个**沙尘概率积分值**，表示该视线上有多少沙尘。

---

### 2.2 位置编码

原始 xyz 坐标直接作为 MLP 输入时，网络难以拟合高频细节。
NeRF 论文（Mildenhall et al., 2020）提出用 **Fourier 特征**扩展输入：

```
γ(p) = [ p,  sin(2⁰π·p), cos(2⁰π·p),
              sin(2¹π·p), cos(2¹π·p),
              …
              sin(2^(L-1)·π·p), cos(2^(L-1)·π·p) ]
```

- 对坐标 (x,y,z)：L = 10 → 输出维度 = 3 + 2×10×3 = **63**
- 对方向 (dx,dy,dz)：L = 4 → 输出维度 = 3 + 2×4×3 = **27**

这极大提升了网络对空间细节的表达能力。

---

### 2.3 分层重要性采样

仅均匀采样效率低（大多数空旷处对颜色贡献极小）。
DustNeRF 使用两级网络（**coarse + fine**）：

**第一级（Coarse）**：在 [t_near, t_far] 均匀采样 `n_coarse=64` 个点，运行网络得到粗略权重 {wᵢ}。

**第二级（Fine）**：把粗略权重归一化为分段概率密度函数（PDF），再按此 PDF **重采样** `n_fine=128` 个点（密集采样高密度区域），与 coarse 点合并后再次运行更精细的网络。

```
PDF(tᵢ) ∝ wᵢ + ε      （ε 防止零权重）
z_fine  ~ PDF           （逆变换采样）
z_all   = sort(z_coarse ∪ z_fine)   （合并并排序）
```

最终 fine 渲染使用 n_coarse + n_fine = **192** 个采样点，精度远高于单纯粗采样。

---

### 2.4 沙尘加权损失

标准 NeRF 对所有像素一视同仁。沙尘体素在画面中占比极小，容易被背景损失淹没。
DustNeRF 用**背景减除残差**计算每像素的沙尘重要性权重：

```
background = median(前 N 帧)          （时间中值背景估计）
residual(p) = |frame(p) − background(p)|.mean(RGB)
weight(p)   = max(0, residual(p) − threshold) / max_residual
```

然后在光度损失中放大高沙尘权重像素的贡献：

```
weighted_MSE = mean( (1 + α · weight) · (pred_rgb − gt_rgb)² )
```

默认 `α = dust_weight_alpha = 5.0`，即高沙尘像素损失权重最多放大 **6 倍**。

另外还有一个**沙尘正则化损失**，鼓励 `dust_prob` 输出与背景残差权重对齐：

```
dust_reg = MSE(dust_map, weight.clamp(0,1))
```

总损失：

```
L = L_coarse + L_fine + 0.1 · L_dust_reg
```

---

## 3. 代码模块详解

```
density2/
├── src/
│   ├── dataset.py       数据层：视频→帧→背景估计→光线
│   ├── model.py         网络层：位置编码 + DustNeRF MLP
│   ├── renderer.py      渲染层：采样 + 体积分 + 分层采样
│   ├── train.py         训练层：优化器、损失、日志、检查点
│   ├── export.py        导出层：密度网格评估 + 打包
│   └── analyze.py       分析层：训练曲线 + 密度统计 + 报告
├── auto_improve.py      规则引擎：分析报告 → 超参改进
├── visualize_local.py   可视化层：本地 Plotly + Matplotlib
├── deploy.sh            远程部署脚本
├── sync_results.sh      结果下载 + 分析 + 改进脚本
└── run.sh               服务器端一键运行入口
```

---

### 3.1 `dataset.py`

**功能**：构建 PyTorch `Dataset`，每个样本 = 一条光线。

**详细流程**：

```
1. 读取 config/info.json（摄像机内参 + 外参）
2. 对每台摄像机：
   a. 从 .mp4 或帧目录提取帧（每 frame_skip=5 帧取一帧，最多 60 帧）
   b. 取前 background_frames=30 帧做时间中值背景估计
   c. 用针孔相机模型生成光线（每像素一条，H×W 条）
   d. 对每帧：
      - BGR→RGB 归一化到 [0,1]
      - 计算像素沙尘权重（背景残差）
3. 拼接所有摄像机所有帧的光线
4. 总光线数 = 6摄像机 × 60帧 × 1080×720像素 ≈ 2.8 亿条（均匀随机打散后按 batch_rays=1024 采样）
```

**关键函数**：

| 函数 | 作用 |
|------|------|
| `load_config()` | 读 JSON 配置 |
| `get_intrinsics()` | 构建 3×3 内参矩阵 K |
| `get_c2w()` | 获取 4×4 相机到世界变换矩阵 |
| `extract_frames()` | 从 mp4 提取帧（OpenCV） |
| `BackgroundEstimator.estimate()` | 时间中值背景估计 |
| `get_rays()` | 针孔模型生成像素光线（世界坐标系） |
| `compute_dust_weight()` | 背景残差 → 沙尘权重 [0,1] |
| `DustDataset.__getitem__()` | 返回 {rays_o, rays_d, rgb, weight} |

**光线生成公式**（`get_rays`）：

```python
# 像素 (i, j) 在相机坐标系的方向：
dirs = [(i - cx) / fl_x,   -(j - cy) / fl_y,   -1.0]  # OpenGL 约定，-Z 向前
# 转到世界坐标：
rays_d = R @ dirs          # R = c2w[:3,:3]
rays_o = c2w[:3, 3]        # 相机光心
```

---

### 3.2 `model.py`

**网络结构**：

```
输入：3D 坐标 (x,y,z) + 视线方向 (dx,dy,dz)
          ↓ 位置编码
       pos_enc: (63,)    dir_enc: (27,)
          ↓
┌─────────────────────────────────┐
│  Position MLP（8层，256神经元）  │
│  第4层有残差跳跃连接             │
│  激活函数：ReLU                  │
└─────────────────────────────────┘
          ↓ 256维特征向量 h
    ┌─────┼──────────────┐
    ↓     ↓              ↓
sigma_head dust_head  feature_head
 (256→1)   (256→1)    (256→256)
softplus   sigmoid        ↓
    ↓         ↓      colour_layer1(256+27→128)→ReLU
    σ≥0    p∈[0,1]   colour_layer2(128→3)→sigmoid
                          ↓
                       RGB∈[0,1]³
```

**三个输出头**：

| 输出 | 激活 | 含义 |
|------|------|------|
| `sigma` | `softplus` | 体积密度（非负，类似吸收系数） |
| `rgb` | `sigmoid` | 该点的发射颜色 |
| `dust_prob` | `sigmoid` | 该点是沙尘（而非背景结构）的概率 |

**为什么需要 `dust_prob` 输出头？**

σ（密度）高的地方可能是场景中的固体背景（墙、设备），也可能是空气中的沙尘颗粒。
`dust_prob` 分支让网络学习**区分**这两类物质，使最终导出的密度体素网格可以只保留沙尘部分。

---

### 3.3 `renderer.py`

**功能**：实现完整的两级（coarse + fine）光线渲染流程。

**`sample_stratified`**：分层随机采样

```python
# 将 [near, far] 均匀分为 n_samples 段
t = linspace(near, far, n_samples)
# 在每段内随机扰动（perturb=True 时）
if perturb:
    z = lower + (upper - lower) * rand()
# 三维采样点
pts = rays_o + rays_d * z  # (B, n_samples, 3)
```

**`sample_pdf`**：基于粗级权重的重要性采样

```python
# 将粗级权重归一化为 PDF，再求 CDF
cdf = cumsum(weights / sum(weights))
# 用逆变换采样从 CDF 中取 n_fine 个点
z_fine = invert_cdf(uniform_samples)
```

**`volume_render`**：体渲染积分（离散版）

```python
α = 1 - exp(-σ * δ)              # 每段不透明度
T = cumprod([1, 1-α₀, 1-α₁, ...]) # 前缀透射率
w = T * α                         # 体渲染权重
C = Σ wᵢ * rgbᵢ                  # 渲染颜色
D = Σ wᵢ * tᵢ                    # 深度图
A = Σ wᵢ                          # 不透明度图
dust_map = Σ wᵢ * dust_probᵢ     # 沙尘密度图（DustNeRF 专有）
```

**`render_rays`**：完整流程（coarse → fine）

```
输入: rays_o(B,3), rays_d(B,3)
  ↓
[Coarse] 均匀采样 64 点 → 网络前向 → 体渲染 → 粗权重
  ↓
[Fine] 用粗权重重采样 128 点 → 合并 192 点 → 网络前向 → 体渲染
  ↓
输出字典:
  coarse/rgb, coarse/depth, coarse/acc, coarse/dust
  fine/rgb,   fine/depth,   fine/acc,   fine/dust
```

---

### 3.4 `train.py`

**训练循环详解**：

```
初始化:
  1. 加载 config/info.json
  2. 创建 DustDataset（加载所有视频帧并生成光线）
  3. 创建 DataLoader（batch_rays=1024，shuffle=True）
  4. 创建 coarse 和 fine 两个 DustNeRF 模型（共享优化器）
  5. Adam 优化器，初始 lr=5e-4
  6. 指数衰减学习率调度器（250000步衰减到原来的10%）
  7. 若 --resume，从 checkpoints/ckpt_latest.pt 恢复
  8. 初始化 TensorBoard SummaryWriter
  9. 初始化 JsonlLogger（写入 train_metrics.jsonl，供离线分析）

每个训练步:
  1. 取一批光线 (rays_o, rays_d, gt_rgb, weight)
  2. 调用 render_rays → 得到 {coarse/rgb, fine/rgb, fine/dust}
  3. 计算损失:
       L_c = weighted_MSE(coarse_rgb, gt_rgb, weight, α)
       L_f = weighted_MSE(fine_rgb,   gt_rgb, weight, α)
       L_dust = MSE(fine_dust, weight.clamp(0,1))
       L = L_c + L_f + 0.1 * L_dust
  4. 反向传播 + 梯度裁剪（max_norm=1.0）
  5. 优化器步进 + 学习率调度器步进
  6. 每 log_every=500 步：打印日志、写 TensorBoard、写 JSONL
  7. 每 save_every=10000 步：保存检查点
```

**检查点内容**：

```json
{
  "step": 50000,
  "coarse": { "pos_layers.0.weight": [...], ... },
  "fine":   { "pos_layers.0.weight": [...], ... },
  "optimizer": { "state": {...}, "param_groups": [...] },
  "scheduler": { "last_epoch": 50000, "_last_lr": [...] }
}
```

**JSONL 训练日志**（每 500 步一行）：

```json
{"step": 500, "loss": 0.0842, "loss_c": 0.0421, "loss_f": 0.0401, "loss_dust": 0.0200, "psnr": 18.53, "lr": 4.999e-4, "elapsed_s": 42.1}
{"step": 1000, "loss": 0.0721, ...}
```

---

### 3.5 `export.py`

**功能**：训练完成后，用 fine 模型在规则三维体素网格上评估密度，并打包所有输出。

**流程**：

```
1. 从检查点加载 fine 模型
2. 从配置中摄像机位置计算场景边界（所有相机光心±2m）
3. 在 128³ 网格上均匀采样，批次评估 fine 模型（batch=8192，方向统一取 (0,0,-1)）
4. 保存:
   a. density_grid.npz   NumPy 压缩文件（density, dust_prob, grid_min, grid_max）
   b. density_grid.vtk   VTK 格式（可用 ParaView 打开）
   c. cameras.json       所有摄像机位置和内参
   d. dust_cloud.html    Plotly 交互三维散点图
   e. dust_density_slices.png  XYZ 三个中截面图
5. 打包为 dust_export.tar.gz
```

---

### 3.6 `analyze.py`

**功能**：读取 `density_grid.npz` + `train_metrics.jsonl`，生成分析报告和图表。

**训练曲线分析**：
- 检测**发散**：最后10%步骤损失单调上升
- 检测**平台期**：最后20%步骤 PSNR 提升 < 0.5 dB
- 检测**低质量**：最终 PSNR < 20 dB
- 检测**快速初始收敛**

**密度网格分析**：
- 沙尘覆盖率（>50% 概率体素 / 总体素）
- 密度统计（均值、最大值、P95、P99）
- 稀疏度
- 沙尘云质心和展布方差

**输出**：
- `analysis_report.json` — 所有数值指标 + 文字改进建议
- `training_curves.png` — 损失/PSNR/LR 曲线
- `density_histogram.png` — 密度分布直方图
- `dust_coverage_map.png` — Z 轴最大投影俯视图

---

### 3.7 `auto_improve.py`

**功能**：读取 `analysis_report.json`，用规则引擎自动更新 `config/info.json`。

**规则表**：

| 检测到的问题 | 修改字段 | 变更逻辑 |
|-------------|----------|---------|
| 训练发散 | `learning_rate` | × `LR_DIVERGE_FACTOR`（0.5） |
| PSNR < 20 dB | `max_steps` | × `STEPS_LOW_PSNR_FACTOR`（1.5） |
| 平台期 | `max_steps` | × `STEPS_PLATEAU_FACTOR`（1.3） |
| 平台期 | `lr_decay_factor` | 设为 `LR_DECAY_SLOW`（0.05） |
| 沙尘覆盖 < 0.01% | `dust_threshold` | × `DUST_THRESHOLD_LOW_FACTOR`（0.5） |
| 沙尘覆盖 < 0.01% | `background_frames` | 减 10 |
| 沙尘覆盖 > 30% | `dust_threshold` | × `DUST_THRESHOLD_HIGH_FACTOR`（1.5） |
| 沙尘覆盖 > 30% | `background_frames` | 加 15 |
| 体积几乎全空 | `n_samples_coarse/fine` | × `SAMPLES_SPARSE_FACTOR`（2） |
| `dust_prob` 均值 < 0.05 | `dust_weight_alpha` | × `DUST_ALPHA_FACTOR`（2） |
| PSNR ≥ 25 dB | `export.grid_resolution` | 设为 `EXPORT_RES_HIGH`（256） |

改变写入前先备份 `config/info.json` → `config/info.json.bak`。

---

### 3.8 `visualize_local.py`

**功能**：在本地机器上可视化下载回来的 export 数据。

**功能特性**：
- `plot_3d_cloud()` — Plotly 交互三维散点图（颜色=沙尘概率，相机位置用菱形标注）
- `plot_slices()` — matplotlib X/Y/Z 中截面密度图
- `print_summary()` — 打印密度统计摘要（沙尘体积、覆盖率等）

---

## 4. 完整运行流程

### 4.1 数据准备

将各摄像机视频放入 `data/` 目录，命名规则：

```
data/
  train00.mp4    # 0° 摄像机
  train01.mp4    # 60° 摄像机
  train02.mp4    # 120° 摄像机
  train03.mp4    # 180° 摄像机
  train04.mp4    # 240° 摄像机
  train05.mp4    # 300° 摄像机
```

也支持预提取帧目录（`data/train00/frame_0000.png`，...）。

如果你的摄像机参数与默认值不同，编辑 `config/info.json` 中每个摄像机的 `fl_x`、`fl_y`、`cx`、`cy`、`w`、`h`、`transform_matrix`。

---

### 4.2 本地单机运行

**安装依赖**：

```bash
pip install -r requirements.txt
```

**一键训练 + 导出**：

```bash
bash run.sh all
```

**分步运行**：

```bash
# 仅训练
bash run.sh train

# 断点续训
bash run.sh train --resume

# 仅导出（训练已完成后）
bash run.sh export
```

**自定义路径**：

```bash
CONFIG=config/info.json  DATA=data  OUT=outputs  bash run.sh all
```

**监控训练（TensorBoard）**：

```bash
tensorboard --logdir outputs/logs --host 0.0.0.0 --port 6006
```

**本地可视化**：

```bash
tar -xzf outputs/dust_export.tar.gz
python visualize_local.py --export export/
# 保存到文件（无 GUI）：
python visualize_local.py --export export/ \
    --save-html dust_cloud.html \
    --save-png  dust_slices.png
```

---

### 4.3 远程服务器运行（推荐）

训练需要 GPU，推荐在远程服务器上进行。

**第0步：配置服务器**

```bash
cp server.env.example server.env
# 编辑 server.env，填入服务器 IP、端口、用户名、目录
nano server.env
```

```ini
REMOTE_HOST=192.168.1.100    # 服务器 IP
REMOTE_PORT=22               # SSH 端口（可自定义）
REMOTE_USER=ubuntu           # SSH 用户名
REMOTE_DIR=/home/ubuntu/density2
```

**第1步：一键部署并启动训练**

```bash
bash deploy.sh all
```

`deploy.sh` 完成以下步骤：
1. `rsync` 上传所有代码和数据到服务器
2. 在服务器上创建 Python venv 并安装 requirements.txt
3. 用 `nohup bash run.sh train` 在后台启动训练（进程 PID 写入 `.train_pid`）

**查看训练进度**：

```bash
bash deploy.sh status
# 或直接 SSH：
ssh -p 22 ubuntu@192.168.1.100 tail -f /home/ubuntu/density2/train.log
```

---

### 4.4 下载与分析

**训练完成后下载并分析**（自动触发导出）：

```bash
# 等待训练完成后自动下载（阻塞等待）
bash sync_results.sh --wait

# 或立即下载（训练已完成）
bash sync_results.sh
```

`sync_results.sh` 自动完成：
1. 轮询服务器进程（`--wait` 模式，每 60 秒检查一次）
2. 如果没有导出包则自动触发 `bash run.sh export`
3. `scp` 下载 `dust_export.tar.gz` + `train_metrics.jsonl` + `train.log`
4. 解压到 `results/export/`
5. 调用 `python src/analyze.py` → 生成 `results/analysis/`
6. 调用 `python auto_improve.py` → 更新 `config/info.json`

**查看分析结果**：

```bash
# 分析报告（含改进建议）
cat results/analysis/analysis_report.json

# 交互式三维可视化（在浏览器打开）
open results/export/dust_cloud.html

# 完整交互式可视化
python visualize_local.py --export results/export/
```

---

### 4.5 自动改进后重训

**查看建议的改动**（不实际写入）：

```bash
python auto_improve.py \
    --report results/analysis/analysis_report.json \
    --dry-run
```

**应用改动**：

```bash
python auto_improve.py \
    --report results/analysis/analysis_report.json
# → 自动备份 config/info.json.bak，写入改进后的 config/info.json
```

**使用改进后的配置重训**：

```bash
bash deploy.sh train    # 上传新 config，服务器断点续训
bash sync_results.sh --wait  # 再次等待、下载、分析
```

重复上述循环，直到 PSNR ≥ 25 dB、沙尘覆盖合理。

---

## 5. 配置文件详解

完整配置文件 `config/info.json` 的所有字段说明：

### `scene`（元信息）

```json
{
  "description": "Six-camera dust density reconstruction scene",
  "camera_count": 6,
  "camera_ids": ["train00", ..., "train05"],
  "camera_spacing_deg": 60,
  "video_dir": "data",
  "output_dir": "outputs"
}
```

### `cameras[]`（每台摄像机）

| 字段 | 说明 | 单位 |
|------|------|------|
| `id` | 摄像机名（对应视频文件名前缀） | — |
| `angle_deg` | 水平角度 | 度 |
| `fl_x` / `fl_y` | 焦距 | 像素 |
| `cx` / `cy` | 主点（光轴与像面交点） | 像素 |
| `w` / `h` | 图像宽 / 高 | 像素 |
| `transform_matrix` | 4×4 相机到世界变换矩阵（NeRF 约定：+Y 向上，-Z 向前） | — |

### `training`（训练超参数）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `near` / `far` | 0.1 / 10.0 | 光线近/远裁剪距离（米） |
| `n_samples_coarse` | 64 | 粗级每条光线采样点数 |
| `n_samples_fine` | 128 | 精级重采样点数 |
| `batch_rays` | 1024 | 每步训练光线批大小 |
| `learning_rate` | 5e-4 | Adam 初始学习率 |
| `lr_decay_steps` | 250000 | 学习率指数衰减的步数跨度 |
| `lr_decay_factor` | 0.1 | 学习率总衰减倍数（最终 = 初始 × 0.1） |
| `max_steps` | 200000 | 总训练步数 |
| `save_every` | 10000 | 检查点保存间隔 |
| `log_every` | 500 | 日志记录间隔 |
| `frames_per_video` | 60 | 每路视频最多提取的帧数 |
| `frame_skip` | 5 | 每隔 N 帧取一帧 |
| `background_frames` | 30 | 用于背景估计的帧数 |
| `dust_threshold` | 0.05 | 背景残差低于此值不计为沙尘 |
| `dust_weight_alpha` | 5.0 | 沙尘像素损失权重放大倍数 |

### `export`（导出设置）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `grid_resolution` | 128 | 密度网格每轴体素数（总体素数 = resolution³） |
| `density_threshold` | 0.01 | HTML 可视化的密度阈值 |
| `output_formats` | `["npz","vtk","html"]` | 导出格式 |

---

## 6. 输出文件说明

训练后所有输出均在 `outputs/` 目录下：

```
outputs/
├── checkpoints/
│   ├── ckpt_0010000.pt      # 每 10000 步保存一次
│   ├── ckpt_0020000.pt
│   ├── ...
│   └── ckpt_latest.pt       # 最新检查点（软链接/副本）
├── logs/
│   └── events.out.tfevents.*  # TensorBoard 日志
├── train_metrics.jsonl        # 每 500 步一行的训练指标（JSONL）
├── export/
│   ├── density_grid.npz       # NumPy 压缩密度体（density + dust_prob）
│   ├── density_grid.vtk       # VTK 格式（ParaView 可读）
│   ├── cameras.json           # 所有摄像机位置和内参
│   ├── dust_cloud.html        # Plotly 交互三维图
│   └── dust_density_slices.png  # 三个方向的中截面图
└── dust_export.tar.gz         # 上述 export/ 的打包压缩文件
```

下载后分析结果在 `results/` 目录：

```
results/
├── dust_export.tar.gz         # 从服务器下载的压缩包
├── train_metrics.jsonl        # 训练指标日志
├── train.log                  # 训练控制台日志
├── export/                    # 解压后的导出文件
└── analysis/
    ├── analysis_report.json   # 结构化分析报告（含改进建议）
    ├── training_curves.png    # 损失 / PSNR / LR 曲线图
    ├── density_histogram.png  # 密度分布直方图
    └── dust_coverage_map.png  # 俯视沙尘覆盖图
```

---

## 7. 数据流程总览图

```
┌─────────────────────────────────────────────────────────────────────┐
│  INPUT                                                              │
│  train00.mp4 ~ train05.mp4  (6 路摄像机视频)                        │
│  config/info.json           (摄像机内外参 + 训练配置)               │
└────────────────────────────────┬────────────────────────────────────┘
                                 │  dataset.py
                    ┌────────────▼────────────┐
                    │  帧提取 (每5帧取1帧)      │
                    │  背景估计 (时间中值)       │
                    │  光线生成 (针孔模型)       │
                    │  沙尘权重 (背景残差)       │
                    │  → DustDataset           │
                    │    rays_o / rays_d        │
                    │    rgb / weight           │
                    └────────────┬────────────┘
                                 │  DataLoader (batch=1024)
                    ┌────────────▼────────────┐
                    │  训练循环 train.py        │
                    │                          │
                    │  ┌──────────────────┐   │
                    │  │ render_rays()     │   │
                    │  │ [coarse] 64采样  │   │
                    │  │   ↓ DustNeRF     │   │
                    │  │   ↓ 体渲染       │   │
                    │  │   ↓ 权重         │   │
                    │  │ [fine] 128重采样 │   │
                    │  │   ↓ DustNeRF     │   │
                    │  │   ↓ 体渲染       │   │
                    │  └──────────────────┘   │
                    │  ↓                       │
                    │  L = L_c + L_f + 0.1·L_d│
                    │  ↓ Adam + 梯度裁剪       │
                    │  ↓ 保存检查点/日志        │
                    └────────────┬────────────┘
                                 │  export.py
                    ┌────────────▼────────────┐
                    │  在 128³ 网格评估 fine   │
                    │  density_grid.npz        │
                    │  density_grid.vtk        │
                    │  dust_cloud.html         │
                    │  → dust_export.tar.gz    │
                    └────────────┬────────────┘
                                 │  scp 下载
                    ┌────────────▼────────────┐
                    │  analyze.py              │
                    │  训练曲线分析             │
                    │  密度体素统计             │
                    │  → analysis_report.json  │
                    │  → 图表 PNG              │
                    └────────────┬────────────┘
                                 │  auto_improve.py
                    ┌────────────▼────────────┐
                    │  规则引擎                 │
                    │  → 更新 config/info.json │
                    │  → 备份 .bak             │
                    └────────────┬────────────┘
                                 │  下一轮训练
                                 └──────────►  (循环)
```

---

## 8. 常见问题与调参指南

### Q1：训练时 PSNR 始终低于 15 dB

**可能原因及解决方法**：

| 原因 | 解决 |
|------|------|
| 摄像机内外参不准确 | 重新标定，更新 `config/info.json` |
| 训练步数不足 | 增大 `max_steps`（如 300000~500000） |
| 学习率过高导致震荡 | 将 `learning_rate` 降至 `2e-4` |
| `near`/`far` 范围不合适 | 根据实际场景尺寸调整 |
| 背景估计帧数不足 | 增大 `background_frames` |

### Q2：沙尘覆盖率异常低（<0.01%）

- 降低 `dust_threshold`（如 0.02~0.01）
- 减少 `background_frames`（背景可能包含了沙尘帧）
- 确认视频确实拍到了沙尘（不是空场景）

### Q3：沙尘覆盖率异常高（>30%）

- 提高 `dust_threshold`（如 0.08~0.15）
- 增大 `background_frames`（背景估计更稳健）
- 检查背景是否有移动物体被误检为沙尘

### Q4：训练发散（loss 越来越大）

- 将 `learning_rate` 减半（如 2.5e-4）
- 减小 `batch_rays`（降低梯度方差）
- 检查 GPU 显存是否不足（OOM 会导致异常值）

### Q5：GPU 显存不足（OOM）

- 减小 `batch_rays`（如 512）
- 减小 `n_samples_coarse` 和 `n_samples_fine`（如 48 + 96）

### Q6：平台期 - 训练已收敛但 PSNR 不再提升

- 增大 `max_steps`（多训练）
- 降低 `lr_decay_factor`（如 0.05）使学习率衰减更慢
- 增大 `dust_weight_alpha`（让沙尘像素获得更高损失权重）

### Q7：如何手动运行分析而不通过 sync_results.sh

```bash
# 单独运行分析
python src/analyze.py \
    --export  results/export/ \
    --metrics results/train_metrics.jsonl \
    --out     results/analysis/

# 单独运行自动改进（干运行查看将要做的改变）
python auto_improve.py \
    --report results/analysis/analysis_report.json \
    --dry-run

# 应用改进
python auto_improve.py \
    --report results/analysis/analysis_report.json
```

### Q8：如何只做可视化而不重训

```bash
# 打开交互三维图（浏览器）
python visualize_local.py --export results/export/

# 保存到文件（无 GUI 服务器）
python visualize_local.py --export results/export/ \
    --save-html results/dust_cloud.html \
    --save-png  results/dust_slices.png
```

---

*文档生成时间：2026-03  |  对应仓库：`zjsygqsdks/density2`*
