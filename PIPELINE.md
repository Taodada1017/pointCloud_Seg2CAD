# PointCloud2CAD 流程文档

> **基于开放词汇语义分割的室内三维点云语义实例分割与 CAD 平面图生成**
>
> 本文档完整介绍三个阶段的串联流程、数据格式与流转方式。

---

## 目录

- [1. 系统总览](#1-系统总览)
- [2. 端到端数据流](#2-端到端数据流)
- [3. 阶段一：2D 开放词汇语义分割（rma-dino-sam）](#3-阶段一2d-开放词汇语义分割rma-dino-sam)
  - [3.1 功能概述](#31-功能概述)
  - [3.2 输入](#32-输入)
  - [3.3 处理流程](#33-处理流程)
  - [3.4 输出](#34-输出)
  - [3.5 运行命令](#35-运行命令)
- [4. 阶段二：3D 语义实例融合（PointCloud_Segement_v0）](#4-阶段二3d-语义实例融合pointcloud_segement_v0)
  - [4.1 功能概述](#41-功能概述)
  - [4.2 输入](#42-输入)
  - [4.3 处理流程](#43-处理流程)
  - [4.4 输出](#44-输出)
  - [4.5 运行命令](#45-运行命令)
- [5. 阶段二→三 衔接（stage2_to_stage3.py）](#5-阶段二三-衔接stage2_to_stage3py)
  - [5.1 为什么需要衔接](#51-为什么需要衔接)
  - [5.2 衔接逻辑](#52-衔接逻辑)
  - [5.3 运行命令](#53-运行命令)
- [6. 阶段三：点云→2D 线段提取（LiDAR2BIM-Registration 子模块）](#6-阶段三点云2d-线段提取lidar2bim-registration-子模块)
  - [6.1 功能概述](#61-功能概述)
  - [6.2 输入](#62-输入)
  - [6.3 处理流程](#63-处理流程)
  - [6.4 输出](#64-输出)
- [7. 完整数据流转图](#7-完整数据流转图)
- [8. 目录结构与文件索引](#8-目录结构与文件索引)
- [9. 快速上手](#9-快速上手)

---

## 1. 系统总览

本项目实现了一套 **从 LiDAR 扫描点云到 2D CAD 线段** 的自动化处理管线（Pipeline），分为三个串行阶段：

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        PointCloud2CAD Pipeline                             │
│                                                                            │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────────┐  │
│  │   阶段一      │    │    阶段二         │    │      阶段三              │  │
│  │ 2D 语义分割   │───▶│ 3D 语义实例融合   │───▶│  点云 → 2D 线段提取      │  │
│  │  (Python)     │    │    (C++)         │    │  (Python)                │  │
│  │ rma-dino-sam  │    │ PointCloud_Seg.. │    │ stage2_to_stage3.py      │  │
│  └──────────────┘    └──────────────────┘    └──────────────────────────┘  │
│                                                                            │
│  RGB 图像 ──────▶ 分割 Mask ──┐                                           │
│                               ├──▶ 语义点云 ──▶ 2D 线段 + 平面图图像       │
│  LiDAR 点云 ─────────────────┘                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

| 阶段 | 模块 | 语言 | 核心功能 |
|------|------|------|---------|
| **一** | `rma-dino-sam/` | Python | 用 RAM + GroundingDINO + SAM 对多视角 RGB 图像进行开放词汇语义实例分割 |
| **二** | `PointCloud_Segement_v0/` | C++ | 将 2D 分割结果通过相机参数反投影到 3D 点云，多帧融合生成语义实例点云 |
| **三** | `stage2_to_stage3.py`（调用 LiDAR2BIM-Registration 核心算法） | Python | 从分割后点云中提取墙壁结构，投影到 2D 平面并用 Hough 变换检测线段 |

---

## 2. 端到端数据流

下图展示了数据从原始采集到最终输出的完整流转路径：

```
                    ┌─────────────────────┐
                    │    原始采集数据       │
                    │                     │
                    │  • LiDAR 点云 (.pcd)│
                    │  • RGB 图像序列      │
                    │  • 相机位姿          │
                    │  • 相机内参          │
                    └─────────┬───────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               │
   ┌─────────────────────┐                    │
   │     阶段一            │                    │
   │  2D 语义分割          │                    │
   │                       │                    │
   │  输入: RGB 图像       │                    │
   │  输出: mask + label   │                    │
   └──────────┬────────────┘                    │
              │                               │
              │  *_mask.png (16-bit 实例掩码)  │
              │  *_label.json (语义标签)       │
              │                               │
              ▼                               ▼
   ┌──────────────────────────────────────────────┐
   │               阶段二                          │
   │          3D 语义实例融合                       │
   │                                              │
   │  输入: 点云 + 图像 + mask + 位姿 + 内参       │
   │  输出: 语义实例点云 + 语义标签映射             │
   └──────────────────┬───────────────────────────┘
                      │
                      │  removed.pcd    (背景结构点云)
                      │  segement.pcd   (实例着色点云)
                      │  info.txt       (实例→语义标签)
                      │  points.txt     (实例→点索引)
                      │
                      ▼
   ┌──────────────────────────────────────────────┐
   │         阶段二→三 衔接脚本                     │
   │        stage2_to_stage3.py                    │
   │                                              │
   │  1. 语义分离: 墙壁 / 地面 / 天花板            │
   │  2. 高度裁剪: 保留墙壁高度区间                │
   │  3. 投影: 3D → 2D (z=0 平面)                 │
   │  4. Hough 线段检测                            │
   └──────────────────┬───────────────────────────┘
                      │
                      │  walls_2d.png   (2D 投影图像)
                      │  linesegs.txt   (2D 线段坐标)
                      │  walls.pcd      (墙壁点云)
                      │  ground.pcd     (地面点云)
                      │
                      ▼
             ┌─────────────────┐
             │   最终输出       │
             │                 │
             │  2D 建筑线段     │
             │  (CAD 平面图)   │
             └─────────────────┘
```

---

## 3. 阶段一：2D 开放词汇语义分割（rma-dino-sam）

### 3.1 功能概述

利用三个 2D 基础视觉模型的级联，对 RGB 图像进行**开放词汇**的语义实例分割——不需要预定义类别列表，能识别任意物体。

模型级联关系：

```
RGB 图像
  │
  ▼
┌──────────────────────┐
│  RAM (Recognize       │   "chair, table, wall, floor, door, window, ..."
│  Anything Model)      │ → 自动识别图像中的所有语义标签（文本 tag）
└──────────┬───────────┘
           │ 标签文本 (text prompt)
           ▼
┌──────────────────────┐
│  GroundingDINO        │ → 用文本标签做开放集目标检测
│  (开放集检测器)        │   输出: 每个物体的 bounding box
└──────────┬───────────┘
           │ bounding box
           ▼
┌──────────────────────┐
│  SAM (Segment         │ → 根据 bounding box 提示生成高精度实例掩码
│  Anything Model)      │   输出: 像素级 mask
└──────────────────────┘
```

### 3.2 输入

| 文件 | 格式 | 说明 |
|------|------|------|
| RGB 图像目录 | `.jpg` / `.png` | 多视角相机拍摄的室内场景图像序列 |

默认读取 `rma-dino-sam/assets/` 目录。

### 3.3 处理流程

```
对每张图像:
  ① RAM 提取语义标签 → "chair, table, wall, floor, ..."
  ② GroundingDINO 对每个标签做目标检测 → bounding boxes
  ③ SAM 对每个 box 生成实例掩码 → pixel-wise mask
  ④ 合并所有实例为一张 16-bit mask（每个像素值 = 实例 ID）
  ⑤ 保存 mask + label 映射
```

### 3.4 输出

每张输入图像生成两个文件，存放在输出目录的 `prediction_no_augment/` 子目录下：

| 文件 | 格式 | 内容 |
|------|------|------|
| `{timestamp}_mask.png` | 16-bit PNG | 实例分割掩码，每个像素的值表示实例 ID（0 = 背景） |
| `{timestamp}_label.json` | JSON | 实例 ID → 语义标签的映射 |

**mask.png 示例**（概念示意）：

```
┌──────────────────────────┐
│  0  0  0  0  0  0  0  0  │   0 = 背景
│  0  1  1  1  0  2  2  0  │   1 = "chair" 实例
│  0  1  1  1  0  2  2  0  │   2 = "table" 实例
│  0  0  0  0  0  0  0  0  │   3 = "wall" 实例
│  3  3  3  3  3  3  3  3  │   ...
└──────────────────────────┘
```

**label.json 示例**：

```json
{
  "1": "chair",
  "2": "table",
  "3": "wall"
}
```

### 3.5 运行命令

```bash
cd rma-dino-sam

# 配置权重路径（可选，默认读 ./weights/）
export WEIGHTS_DIR="/path/to/weights"

# 运行
python main.py "<输入图像目录>" "<输出目录>"
```

> ⚠️ 模型权重约 5GB，需要 GPU 运行（推荐 NVIDIA GPU ≥ 8GB 显存）。
> 可在远程 GPU 服务器上运行此阶段，将输出文件传回本地即可。

---

## 4. 阶段二：3D 语义实例融合（PointCloud_Segement_v0）

### 4.1 功能概述

将阶段一的 2D 分割结果反投影到 3D 空间，与 LiDAR 点云融合。核心算法：

1. **体素哈希建图 (VoxelHashMap)**：高效的 3D 空间索引
2. **2D → 3D 反投影**：利用相机内外参，将 2D mask 映射到 3D 点
3. **贝叶斯标签融合**：多帧观测的概率语义标签决策
4. **跨帧实例合并**：基于 IoU 的多帧实例关联，保持 ID 一致性

### 4.2 输入

| 文件/目录 | 来源 | 格式 | 说明 |
|-----------|------|------|------|
| `colorized.pcd` | LiDAR SLAM | PCD (点云) | 原始 SLAM 输出的带颜色 3D 点云 |
| `config.yaml` | 手动配置 | YAML | 相机内参 (fx, fy, cx, cy, w, h) + 算法参数 |
| `transforms.json` | SLAM/SfM | JSON | 每帧图像的相机外参（位姿矩阵） |
| `color/left/*.jpg` | 相机采集 | JPEG | 左相机 RGB 图像序列 |
| `color/right/*.jpg` | 相机采集 | JPEG | 右相机 RGB 图像序列 |
| `prediction_no_augment/` | **阶段一输出** | PNG + JSON | 每帧的 mask 和 label 文件 |

**数据目录结构**：

```
data_folder/
├── colorized.pcd                        ← LiDAR 点云
├── config.yaml                          ← 相机参数 + 算法配置
├── transforms.json                      ← 图像位姿
├── color/
│   ├── left/                            ← 左相机图像
│   │   ├── 1765952490650809088.jpg
│   │   ├── 1765952499151191040.jpg
│   │   └── ...
│   └── right/                           ← 右相机图像
│       ├── 1765952490650811904.jpg
│       └── ...
└── prediction_no_augment/               ← ★ 阶段一输出
    ├── 1765952490650809088_mask.png
    ├── 1765952490650809088_label.json
    ├── 1765952499151191040_mask.png
    ├── 1765952499151191040_label.json
    └── ...
```

> **关键串联点**：`prediction_no_augment/` 目录就是阶段一的输出。
> 文件名中的时间戳必须与 `color/` 目录中的图像文件名对应。

### 4.3 处理流程

```
对每帧图像 (按时间序遍历):
  ① 读取 RGB 图像 + 对应的 mask.png + label.json
  ② 从 transforms.json 获取该帧的相机位姿 (4×4 矩阵)
  ③ 对 mask 中每个实例区域:
     a. 根据 2D 像素坐标 + 深度 + 内参 → 反投影到 3D 点
     b. 在体素哈希表中查询匹配的已有实例 (IoU 关联)
     c. 若匹配成功 → 合并到已有实例，贝叶斯更新标签概率
     d. 若无匹配   → 创建新实例
  ④ 定期合并过度分割的实例 (merge_inflation + merge_iou)
  ⑤ 遍历完所有帧后:
     a. 对 SLAM 点云按实例 ID 着色 → segement.pcd
     b. 移除家具实例，仅保留 floor/ceiling/carpet → removed.pcd
     c. 导出实例信息 → info.txt + points.txt
```

### 4.4 输出

所有输出文件写入指定的输出目录：

| 文件 | 格式 | 内容 |
|------|------|------|
| `segement.pcd` | PCD | 按实例 ID 着色的完整 SLAM 点云（每个实例一个颜色） |
| `removed.pcd` | PCD | 剔除家具后的结构性背景点云（保留 floor + ceiling + carpet） |
| `info.txt` | 文本 | 实例语义信息，每行: `实例ID 帧ID 标签1:分数1,标签2:分数2,...` |
| `points.txt` | 二进制 | 每个实例对应原始 SLAM 点云中的点索引 |
| `config.txt` | 文本 | 运行时的配置备份 |

**info.txt 格式示例**：

```
0 42 chair:0.85,furniture:0.12,stool:0.03
1 42 table:0.91,desk:0.06,counter:0.03
2 15 floor:0.97,carpet:0.02,rug:0.01
3 23 wall:0.82,partition:0.10,board:0.08
4 38 ceiling:0.95,roof:0.03,top:0.02
```

每行含义：`[实例ID] [首次出现的帧ID] [标签:置信度,标签:置信度,...]`

**points.txt 二进制格式**：

```
对于每个实例:
  ├── uint32_t  instance_id     (4 字节)
  ├── size_t    point_count     (8 字节, 该实例包含多少个点)
  └── size_t[]  point_indices   (point_count × 8 字节, 每个点在原始点云中的下标)
```

### 4.5 运行命令

```bash
cd PointCloud_Segement_v0/build
./Test_main "<数据目录>" "<输出目录>"

# 示例
./Test_main ../test_data ../test_data/outputs/run01
```

---

## 5. 阶段二→三 衔接（stage2_to_stage3.py）

### 5.1 为什么需要衔接

阶段二和阶段三之间存在**数据格式差异**，不能直接对接：

| | 阶段二输出 | 阶段三需要 |
|---|---|---|
| **墙壁点云** | 混合在 `removed.pcd` 中 | 独立的 `walls.pcd` |
| **地面点云** | 混合在 `removed.pcd` 中 | 独立的 `ground.pcd` |
| **语义标签** | `info.txt` + `points.txt`（需解析） | 已按语义分好的独立文件 |

具体来说：

- **阶段二**的 `removed.pcd` 包含墙壁 + 地板 + 天花板的**混合**点云，语义信息存储在 `info.txt`（标签）和 `points.txt`（点索引）两个独立文件中
- **阶段三**的 `PCD2d` 类期望接收**已分离的** `walls_pcd` 和 `ground_pcd` 两个独立点云对象

`stage2_to_stage3.py` 负责填补这个缺口。

### 5.2 衔接逻辑

```
┌───────────────────────────────────────────────────────────────┐
│                   stage2_to_stage3.py                          │
│                                                               │
│   输入:                                                       │
│   ├── colorized.pcd (原始 SLAM 点云，带颜色)                  │
│   ├── info.txt      (实例 → 语义标签映射)                     │
│   └── points.txt    (实例 → 点索引，二进制)                    │
│                                                               │
│   步骤 ①: 解析 info.txt                                      │
│   ┌──────────────────────────────────────────────────┐        │
│   │ 实例 0 → chair:0.85 → 家具 (排除)               │        │
│   │ 实例 1 → table:0.91 → 家具 (排除)               │        │
│   │ 实例 2 → floor:0.97 → ★ 地面                    │        │
│   │ 实例 3 → wall:0.82  → ★ 墙壁                    │        │
│   │ 实例 4 → ceiling:0.95 → 天花板 (排除)            │        │
│   └──────────────────────────────────────────────────┘        │
│                                                               │
│   步骤 ②: 解析 points.txt (二进制)                            │
│   ┌──────────────────────────────────────────────────┐        │
│   │ 实例 2 → [点下标 102, 103, 104, 105, ...]       │        │
│   │ 实例 3 → [点下标 500, 501, 502, 503, ...]       │        │
│   └──────────────────────────────────────────────────┘        │
│                                                               │
│   步骤 ③: 从 colorized.pcd 中按下标提取                       │
│   ┌─────────────────┐  ┌─────────────────┐                   │
│   │  ground.pcd     │  │   walls.pcd     │                   │
│   │  (floor/carpet  │  │  (wall/其他     │                   │
│   │   的点, 带颜色) │  │   结构, 带颜色) │                   │
│   └────────┬────────┘  └────────┬────────┘                   │
│            │                    │                             │
│   步骤 ④: 墙壁点云 → 2D 线段                                 │
│            │                    │                             │
│            │         ┌──────────▼──────────┐                  │
│            │         │ 高度裁剪            │                  │
│            │         │ 0.2m < z < 1.9m     │                  │
│            │         └──────────┬──────────┘                  │
│            │                    │                             │
│            │         ┌──────────▼──────────┐                  │
│            │         │ 投影到 z=0 平面      │                  │
│            │         │ 3D → 2D             │                  │
│            │         └──────────┬──────────┘                  │
│            │                    │                             │
│            │         ┌──────────▼──────────┐                  │
│            │         │ Points2Image        │                  │
│            │         │ 生成二值图像        │                  │
│            │         │ (scale=60 像素/米)  │                  │
│            │         └──────────┬──────────┘                  │
│            │                    │                             │
│            │         ┌──────────▼──────────┐                  │
│            │         │ HoughLinesP         │                  │
│            │         │ 检测直线段          │                  │
│            │         └──────────┬──────────┘                  │
│            │                    │                             │
│   输出:    │                    │                             │
│   ├── ground.pcd    ◄───────────                              │
│   ├── walls.pcd                                               │
│   ├── walls_2d.png  (投影二值图像)                             │
│   └── linesegs.txt  (每行: x1 y1 x2 y2，物理坐标，米)        │
└───────────────────────────────────────────────────────────────┘
```

### 5.3 运行命令

**完整模式**（有阶段二全部输出时）：

```bash
python stage2_to_stage3.py <阶段二输出目录> \
    --slam <colorized.pcd路径> \
    --output <输出目录>

# 示例
python stage2_to_stage3.py \
    PointCloud_Segement_v0/test_data/outputs/run01 \
    --slam PointCloud_Segement_v0/test_data/colorized.pcd \
    --output pipeline_output
```

**简化模式**（直接从单个点云生成线段，不需要语义分离）：

```bash
python stage2_to_stage3.py --simple <点云.pcd> --output <输出目录>

# 示例: 直接用 removed.pcd
python stage2_to_stage3.py \
    --simple PointCloud_Segement_v0/test_data/outputs/run01/removed.pcd \
    --output pipeline_output
```

---

## 6. 阶段三：点云→2D 线段提取（LiDAR2BIM-Registration 子模块）

### 6.1 功能概述

> 注意：本项目仅使用 LiDAR2BIM-Registration 中 **点云→2D 图像→线段检测** 这一子模块，
> 不使用其完整的 BIM 配准功能。核心算法已内联到 `stage2_to_stage3.py` 中。

核心处理链路：

```
3D 墙壁点云 (Nx3)
    │
    ▼ 高度裁剪
    │ 只保留 min_height ~ max_height (默认 0.2m ~ 1.9m)
    │ 排除地面附近和天花板附近的点
    │
    ▼ Z 轴投影 (flatten)
    │ 所有点的 z 坐标设为 0
    │ 3D 点云 → 2D 点集
    │
    ▼ Points2Image
    │ 将 2D 点集栅格化为二值图像
    │ 分辨率 = scale 像素/米 (默认 60)
    │ 膨胀操作 (dilate_kernel=3) 连接相邻点
    │
    ▼ HoughLinesP
    │ OpenCV 概率 Hough 变换
    │ 检测图像中的直线段
    │
    ▼ 坐标反变换
      像素坐标 → 物理坐标 (米)
      输出线段: [(x1,y1,x2,y2), ...]
```

### 6.2 输入

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 3D 点云 | — | 墙壁点云（numpy Nx3），Z 轴需大致朝上 |
| `min_height` | 0.2 m | 裁剪下界 |
| `max_height` | 1.9 m | 裁剪上界 |
| `scale` | 60 | 投影分辨率（像素/米） |
| `dilate_kernel` | 3 | 形态学膨胀核大小 |
| `hough_threshold` | 60 | Hough 累加器阈值 |
| `hough_min_length` | 30 | 最小线段长度（像素） |
| `hough_max_gap` | 30 | 线段最大间隙（像素） |

### 6.3 处理流程

以下是 `Points2Image` 的核心数学：

```
给定 N 个 3D 点 P = {(x_i, y_i, z_i)}:

1. 高度裁剪:
   P' = {p ∈ P | min_height ≤ z ≤ max_height}

2. 投影 (flatten):
   P'' = {(x_i, y_i, 0) | (x_i, y_i, z_i) ∈ P'}

3. 平移到非负坐标:
   x_min = min(x_i), y_min = min(y_i)
   P''' = {(x_i - x_min, y_i - y_min) | ...}

4. 栅格化 (Points2Image):
   图像大小: W = ⌈(x_max - x_min) × scale⌉, H = ⌈(y_max - y_min) × scale⌉
   对每个点: img[int(y × scale)][int(x × scale)] = 255

5. 形态学膨胀:
   kernel = dilate_kernel × dilate_kernel
   img = cv2.dilate(img, kernel)

6. Hough 线段检测:
   lines = cv2.HoughLinesP(img, ρ=1, θ=π/180, threshold, minLength, maxGap)

7. 像素 → 物理坐标反变换:
   x_phys = x_pixel / scale + x_min
   y_phys = y_pixel / scale + y_min
```

### 6.4 输出

| 文件 | 格式 | 内容 |
|------|------|------|
| `walls_2d.png` | PNG 图像 | 墙壁点云投影到水平面的二值图像 |
| `linesegs.txt` | 文本 | 检测到的 2D 线段，每行 `x1 y1 x2 y2`（单位：米） |

---

## 7. 完整数据流转图

以下是三个阶段之间所有文件的流转关系汇总：

```
═══════════════════════════════════════════════════════════════════════
              文件级数据流转 (File-level Data Flow)
═══════════════════════════════════════════════════════════════════════

[采集设备]
    │
    ├──→ color/left/*.jpg          ──┐
    ├──→ color/right/*.jpg          │
    ├──→ colorized.pcd             ──┤──→ 阶段二输入目录
    ├──→ transforms.json           ──┤
    └──→ config.yaml               ──┘
              │
              │ (RGB 图像)
              ▼
═══════════ 阶段一 (rma-dino-sam) ════════════════════════════════════

    color/left/1765952490650809088.jpg
    color/right/1765952490650811904.jpg
              │
              ▼
    prediction_no_augment/
    ├── 1765952490650809088_mask.png      ← 16-bit 实例掩码
    ├── 1765952490650809088_label.json    ← {实例ID: "语义标签"}
    ├── 1765952490650811904_mask.png
    ├── 1765952490650811904_label.json
    └── ...
              │
              │ (与 color/ 放入同一数据目录)
              ▼
═══════════ 阶段二 (PointCloud_Segement_v0) ══════════════════════════

    输入: colorized.pcd + config.yaml + transforms.json
        + color/left/ + color/right/
        + prediction_no_augment/
              │
              ▼
    outputs/
    ├── segement.pcd       ← 全部点云 (按实例 ID 着色)
    ├── removed.pcd        ← 背景结构点云 (墙+地+天花板)
    ├── info.txt           ← "0 42 chair:0.85,furniture:0.12"
    ├── points.txt         ← 二进制: [id][count][index,index,...]
    └── config.txt         ← 运行配置备份
              │
              ▼
═══════════ 衔接脚本 (stage2_to_stage3.py) ═══════════════════════════

    输入: colorized.pcd + outputs/info.txt + outputs/points.txt
    (或简化模式: 仅 removed.pcd)
              │
              ▼
    pipeline_output/
    ├── walls.pcd          ← 分离后的墙壁点云 (带颜色)
    ├── ground.pcd         ← 分离后的地面点云
    ├── walls_2d.png       ← 2D 投影二值图像
    └── linesegs.txt       ← 2D 线段: "x1 y1 x2 y2" (米)
              │
              ▼
═══════════ 最终结果 ═════════════════════════════════════════════════

    linesegs.txt → 可导入 CAD 软件 / 进一步处理
    walls_2d.png → 可视化参考
```

---

## 8. 目录结构与文件索引

```
PointCloud2CAD/
│
├── rma-dino-sam/                     # ===== 阶段一 =====
│   ├── main.py                       # 主入口: python main.py <input> <output>
│   ├── assets/                       # 默认输入目录 (放入 RGB 图像)
│   ├── weights/                      # 模型权重 (或通过 WEIGHTS_DIR 环境变量指定)
│   │   ├── ram_swin_large_14m.pth            (370MB)
│   │   ├── groundingdino_swint_ogc.pth       (694MB)
│   │   └── sam_vit_h_4b8939.pth              (2.4GB)
│   ├── recognize-anything/           # RAM 模型
│   ├── GroundingDINO/                # GroundingDINO 模型
│   ├── segment_anything/             # SAM 模型
│   └── clip/                         # CLIP 模型
│
├── PointCloud_Segement_v0/           # ===== 阶段二 =====
│   ├── main.cpp                      # 主入口: ./Test_main <data_dir> <output_dir>
│   ├── src/
│   │   └── mapping/
│   │       ├── SemanticMapping.cpp    # 核心: 投影融合 + 实例管理
│   │       ├── SemanticDict.h         # 语义类别定义 (floor/ceiling/carpet)
│   │       └── Instance.h             # 实例数据结构
│   ├── cmake/                        # CMake 依赖配置
│   └── test_data/                    # 示例数据
│       ├── colorized.pcd             # SLAM 点云
│       ├── config.yaml               # 配置文件
│       ├── transforms.json           # 位姿信息
│       ├── color/left/*.jpg          # 左相机图像
│       ├── color/right/*.jpg         # 右相机图像
│       └── prediction_no_augment/    # 阶段一输出
│
├── LiDAR2BIM-Registration/           # ===== 阶段三参考代码 =====
│   ├── preprocess/
│   │   ├── geometry/
│   │   │   ├── pc2img.py             # Points2Image 核心投影算法
│   │   │   └── point_cloud.py        # PCD2d 点云处理
│   │   └── datasets/
│   │       ├── lineseg_manager.py    # Hough 线段检测
│   │       └── pcd2d_manager.py      # 点云→2D 管理
│   └── docs/
│       └── prepare.md                # 数据准备说明
│
├── stage2_to_stage3.py               # ===== 衔接脚本 =====
│                                     # 串联阶段二→三，语义分离 + 线段提取
│                                     # 核心算法内联，不依赖阶段三 import
│
├── docs/
│   └── PIPELINE.md                   # ← 本文档
├── README.md                         # 项目简介
└── INSTALL.md                        # 环境部署指南
```

---

## 9. 快速上手

### 前提条件

| 阶段 | 环境 | 备注 |
|------|------|------|
| 阶段一 | Python 3.11 + PyTorch + GPU | 可在远程服务器运行 |
| 阶段二 | C++ (CMake + Open3D + Eigen + OpenCV) | 需要编译 |
| 衔接脚本 | Python 3.8+ (`open3d`, `opencv-python`, `numpy`, `pillow`) | 轻量依赖 |

### 一步步运行

```bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: 阶段一 — 2D 语义分割 (GPU 服务器)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cd rma-dino-sam
export WEIGHTS_DIR="/path/to/weights"
python main.py "path/to/color_images/" "path/to/output/"

# 输出: prediction_no_augment/ 目录
# 将该目录拷贝到阶段二数据目录下


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: 阶段二 — 3D 语义实例融合
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cd PointCloud_Segement_v0/build
cmake .. && make
./Test_main ../test_data ../test_data/outputs/run01

# 输出: outputs/run01/ 目录
#   ├── segement.pcd
#   ├── removed.pcd
#   ├── info.txt
#   ├── points.txt
#   └── config.txt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: 衔接 + 线段提取
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 方式 A: 完整模式 (语义分离 + 线段提取)
python stage2_to_stage3.py \
    PointCloud_Segement_v0/test_data/outputs/run01 \
    --slam PointCloud_Segement_v0/test_data/colorized.pcd \
    --output pipeline_output

# 方式 B: 简化模式 (直接从一个点云文件生成线段)
python stage2_to_stage3.py \
    --simple PointCloud_Segement_v0/test_data/outputs/run01/removed.pcd \
    --output pipeline_output

# 输出: pipeline_output/ 目录
#   ├── walls.pcd        (仅完整模式)
#   ├── ground.pcd       (仅完整模式)
#   ├── walls_2d.png     (2D 投影图)
#   └── linesegs.txt     (线段坐标，可导入 CAD)
```

### 可调参数速查

| 参数 | 默认值 | 在哪里改 | 影响 |
|------|--------|---------|------|
| 相机内参 (fx, fy, cx, cy) | — | `config.yaml` | 2D→3D 投影精度 |
| `voxel_length` | 0.02 | `config.yaml` | 体素分辨率，越小越精细 |
| `min_det_masks` | 2000 | `config.yaml` | 最小检测像素数，过滤小实例 |
| `merge_iou` | 0.3 | `config.yaml` | 实例合并 IoU 阈值 |
| `bayesian_semantic_likelihood` | `""` | `config.yaml` | 贝叶斯似然矩阵（待补充） |
| `--min-height` | 0.2 | `stage2_to_stage3.py` | 墙壁裁剪下界（米） |
| `--max-height` | 1.9 | `stage2_to_stage3.py` | 墙壁裁剪上界（米） |
| `--scale` | 60 | `stage2_to_stage3.py` | 投影分辨率（像素/米） |
| `--dilate` | 3 | `stage2_to_stage3.py` | 形态学膨胀核大小 |
| `--hough-threshold` | 60 | `stage2_to_stage3.py` | Hough 累加器阈值 |

---

> **文档版本**: 2025-04-21
> **适用代码**: PointCloud2CAD 主分支
