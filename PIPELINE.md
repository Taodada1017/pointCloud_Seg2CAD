# PointCloud2CAD 流程文档

> **基于开放词汇语义分割的室内三维点云语义实例分割与 CAD 平面图生成**
>
> 本文档完整介绍三个阶段的串联流程、数据格式与流转方式。
> **所有描述均以实际代码为准。**

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
- [5. 阶段三：点云 → 2D CAD 线段提取（l2bim）](#5-阶段三点云--2d-cad-线段提取l2bim)
  - [5.1 功能概述](#51-功能概述)
  - [5.2 Step 1：点云投影（C++，pcd_projection）](#52-step-1点云投影cpcd_projection)
  - [5.3 Step 2：直线段检测与规则化（Python，wall_regularization_v2）](#53-step-2直线段检测与规则化pythonwall_regularization_v2)
  - [5.4 编译与运行](#54-编译与运行)
- [6. 完整数据流转图](#6-完整数据流转图)
- [7. 目录结构与文件索引](#7-目录结构与文件索引)
- [8. 快速上手](#8-快速上手)

---

## 1. 系统总览

本项目实现了一套 **从 LiDAR 扫描点云到 2D CAD 平面图** 的自动化处理管线（Pipeline），分为三个串行阶段：

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          PointCloud2CAD Pipeline                                 │
│                                                                                  │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────────────────────┐ │
│  │   阶段一      │    │    阶段二         │    │         阶段三                  │ │
│  │ 2D 语义分割   │───▶│ 3D 语义实例融合   │───▶│  点云 → 2D CAD 线段提取         │ │
│  │  (Python)     │    │    (C++)         │    │  (C++ 投影 + Python 规则化)     │ │
│  │ rma-dino-sam  │    │ PointCloud_Seg.. │    │  l2bim/                        │ │
│  └──────────────┘    └──────────────────┘    └─────────────────────────────────┘ │
│                                                                                  │
│  RGB 图像 ──────▶ 分割 Mask ──┐                                                  │
│                               ├──▶ 语义点云 ──▶ 2D 投影 ──▶ CAD 直线段平面图      │
│  LiDAR 点云 ─────────────────┘                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

| 阶段 | 模块 | 语言 | 核心功能 |
|------|------|------|---------|
| **一** | `rma-dino-sam/` | Python | 用 RAM + GroundingDINO + SAM 对多视角 RGB 图像进行开放词汇语义实例分割 |
| **二** | `PointCloud_Segement_v0/` | C++ | 将 2D 分割结果通过相机参数反投影到 3D 点云，多帧融合生成语义实例点云 |
| **三** | `l2bim/` | C++ + Python | **Step 1** (C++): RANSAC 地面分离、坐标对齐、多切片投票投影、骨架化、Hough 线检测合并；**Step 2** (Python): 曼哈顿对齐、区域分类、Hough 线检测、共线合并、正交拓扑修正 |

---

## 2. 端到端数据流

下图展示了数据从原始采集到最终输出的完整流转路径：

```
                    ┌─────────────────────┐
                    │    原始采集数据       │
                    │                     │
                    │  • LiDAR 点云 (.pcd)│
                    │  • RGB 图像序列      │
                    │    (left/ + right/) │
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
   │  输入: color/left/    │                    │
   │      + color/right/   │                    │
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
   ┌────────────────────────────────────────────┐
   │              阶段三 (l2bim)                 │
   │                                            │
   │  Step 1: pcd_projection (C++)              │
   │    RANSAC 分离 → 坐标对齐 →                 │
   │    多切片投票 → 骨架化 → Hough 合并          │
   │                                            │
   │  Step 2: wall_regularization (Python)      │
   │    曼哈顿对齐 → 区域分类 →                   │
   │    Hough 检测 → 共线合并 → 正交修正          │
   └──────────────────┬─────────────────────────┘
                      │
                      ▼
            ┌─────────────────┐
            │   最终输出       │
            │                 │
            │  projection.png │
            │  CAD 直线段图    │
            │  walls_2d.pcd   │
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
           │ bounding box + NMS (IoU=0.5)
           ▼
┌──────────────────────┐
│  SAM (Segment         │ → 根据 bounding box 提示生成高精度实例掩码
│  Anything Model)      │   输出: 像素级 mask
└──────────────────────┘
```

### 3.2 输入

| 文件 | 格式 | 说明 |
|------|------|------|
| RGB 图像目录 | `.jpg` / `.jpeg` / `.png` / `.bmp` / `.tiff` | 多视角相机拍摄的室内场景图像序列 |

> **代码细节**（`rma-dino-sam/main.py` 第 222 行）：支持的图像格式为 `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`。

输入目录通过命令行参数 `sys.argv[1]` 指定，默认值为 `rma-dino-sam/assets/`。

> ⚠️ **重要**：阶段二的 `IO::construct_sorted_frame_table_3_2()` 函数会同时读取 `color/left/` 和 `color/right/` 两个目录的图像。因此**阶段一必须分别处理左右相机的图像**，确保 `prediction_no_augment/` 中同时包含两个方向的 mask 和 label 文件。

### 3.3 处理流程

```
对每张图像:
  ① RAM 提取语义标签 → "chair, table, wall, floor, ..." (raw_tags)
  ② GroundingDINO 对每个标签做目标检测 → bounding boxes
  ③ NMS 去重 (IoU 阈值 = 0.5，box_threshold = 0.3，text_threshold = 0.25)
  ④ SAM 对每个 box 生成实例掩码 → pixel-wise mask
  ⑤ 合并所有实例为一张 16-bit mask（每个像素值 = 实例 ID，0 = 背景）
  ⑥ 保存 mask + label 映射
```

### 3.4 输出

每张输入图像生成两个文件，存放在输出目录下（**注意：直接输出到 `output_base_dir`，不自动创建 `prediction_no_augment/` 子目录**，需要在命令行中指定输出到 `prediction_no_augment/` 目录以供阶段二读取）：

| 文件 | 格式 | 内容 |
|------|------|------|
| `{image_name}_mask.png` | 16-bit PNG | 实例分割掩码，每个像素的值表示实例 ID（0 = 背景） |
| `{image_name}_label.json` | JSON | 语义标签映射（格式详见下方） |

另外，可视化输出（自动标注的检测结果图）保存在 `output_base_dir/other/` 子目录下。

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

**label.json 实际格式**（基于 `save_mask_data()` 函数，第 139–177 行）：

```json
{
  "raw_tags": "chair . table . wall . floor . ceiling",
  "tags": "chair. table. wall. floor. ceiling",
  "mask": [
    {
      "value": 0,
      "label": "background"
    },
    {
      "value": 1,
      "labels": {"office": 0.42, "chair": 0.42},
      "box": [x1, y1, x2, y2]
    },
    {
      "value": 2,
      "labels": {"table": 0.55},
      "box": [x1, y1, x2, y2]
    }
  ]
}
```

> **格式说明**：
> - `raw_tags`：RAM 模型输出的原始标签字符串
> - `tags`：去重后的单词标签集合，格式为 `"word1. word2. word3"`
> - `mask` 数组：第一个元素固定为 `{value: 0, label: "background"}`，后续每个元素对应一个实例
> - 每个实例包含：`value`（对应 mask 中的像素值 / 实例 ID）、`labels`（单词 → 置信度映射）、`box`（检测框坐标）
> - 标签名称会按空格拆分为多个单词，每个单词独立作为 key，共享同一个置信度分数

### 3.5 运行命令

```bash
cd rma-dino-sam

# 配置权重路径（可选，默认读 ./weights/）
export WEIGHTS_DIR="/path/to/weights"

# 运行（需分别处理左右相机图像）
python main.py "<数据目录>/color/left"  "<数据目录>/prediction_no_augment"
python main.py "<数据目录>/color/right" "<数据目录>/prediction_no_augment"
```

> ⚠️ 模型权重约 3.5GB，推荐 GPU 运行（NVIDIA GPU ≥ 8GB 显存）。
> 代码中 `device` 默认为 `"cpu"`，如需 GPU 加速可修改代码中的 `device = "cuda"`。
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

> **代码细节**（`IO::construct_sorted_frame_table_3_2()` 函数）：该函数同时读取 `data_dir + "/color/left"` 和 `data_dir + "/color/right"` 两个目录的图像，分别排序后与 `transforms.json` 中的帧进行匹配。加载预测结果时使用 `LoadPredictions()` 函数，以图像文件名（不含扩展名）为 key，在 `prediction_no_augment/` 目录下查找 `{name}_mask.png` 和 `{name}_label.json`。

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
    ├── 1765952490650809088_mask.png      ← 对应 color/left/ 中的图像
    ├── 1765952490650809088_label.json
    ├── 1765952490650811904_mask.png      ← 对应 color/right/ 中的图像
    ├── 1765952490650811904_label.json
    └── ...
```

> **关键串联点**：`prediction_no_augment/` 目录就是阶段一的输出。
> 文件名中的时间戳必须与 `color/left/` 和 `color/right/` 目录中的图像文件名对应（不含扩展名）。

### 4.3 处理流程

```
对每帧图像 (按时间序遍历，left 和 right 交替处理):
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

## 5. 阶段三：点云 → 2D CAD 线段提取（l2bim）

### 5.1 功能概述

阶段三是项目的**核心输出阶段**，位于 `l2bim/` 目录。它将三维点云转换为规则化的二维墙体线段图（CAD 平面图）。

处理流程分为两步：

```
┌──────────────────────────────┐      ┌────────────────────────────────────┐
│   Step 1: 点云投影 (C++)      │      │   Step 2: 直线段检测与规则化 (Python)│
│   pcd_projection.cpp          │ ───▶ │   wall_regularization_v2.py         │
│                               │      │                                     │
│ • RANSAC 地面/墙体分离        │      │ • 曼哈顿方向自动对齐                 │
│ • 地面法向量对齐              │      │ • 规则/非规则区域分块判定             │
│ • Yaw 角对齐                  │      │ • Hough 线检测                      │
│ • 多切片投票墙壁投影          │      │ • 共线段合并 (带路径验证)             │
│ • 占用栅格二值化              │      │ • 正交拓扑交叉点修正                 │
│ • 骨架化 (Zhang-Suen/ximgproc)│      │                                     │
│ • Hough 线检测 + 合并         │      │ 输出: CAD 直线段结果图               │
│                               │      │   v2_step8_final_result.png         │
│ 输出: projection.png          │      │                                     │
│   projection_lines_refined.png│      │                                     │
│   walls_2d.pcd                │      │                                     │
└──────────────────────────────┘      └────────────────────────────────────┘
```

> **依赖库**：Step 1 使用 **PCL** + **OpenCV** + **Eigen**；Step 2 使用 **OpenCV** + **NumPy**。详细安装见 [INSTALL.md](INSTALL.md)。

### 5.2 Step 1：点云投影（C++，pcd_projection）

**源文件**：`l2bim/examples/cpp/pcd_projection.cpp`（1133 行）

#### 输入

| 参数 | 说明 |
|------|------|
| `argv[1]` | 输入点云文件（`.pcd` 或 `.ply`）|
| `argv[2]` | 输出目录 |
| `argv[3..29]` | 可选参数（详见下方参数表）|

输入点云可以是：
- 阶段二输出的 `removed.pcd`（剔除家具后的结构点云）
- 或任何室内场景 `.pcd`/`.ply` 文件（如 `l2bim/pcds/` 中的测试数据）

#### 处理流程

```
输入点云 (.pcd/.ply)
    │
    ▼ ① loadCloud()
    │   加载点云文件 (PCL)
    │
    ▼ ② splitGroundWalls()
    │   RANSAC 平面拟合分离地面与墙壁
    │   • distThresh = 0.3m
    │   • maxIter = 1000
    │   • 添加垂直约束: 检测到的平面必须与当前地面法向量的夹角 < 30°
    │   • 输出: ground (地面点云) + walls (墙壁点云) + groundNormal (地面法向量)
    │
    ▼ ③ alignWallsToXYZ()
    │   坐标系对齐 (两阶段):
    │   a. 地面法向量对齐: 旋转使地面法向量朝 Z 轴 → 墙壁与 XY 平面平行
    │   b. Yaw 角对齐: 将墙壁投影到 XY 平面，对投影方向做直方图统计 (1° 分辨率)
    │      找到主方向后旋转，使墙壁尽量沿 X/Y 轴对齐 (曼哈顿世界假设)
    │
    ▼ ④ 高度带过滤
    │   仅保留 lowerZ ≤ z ≤ upperZ 范围内的墙壁点
    │   默认: lowerZ = -1.5m, upperZ = -1.0m (相对于地面高度)
    │
    ▼ ⑤ 多切片投票投影 (Multi-Slice Voting)
    │   将高度范围分为 numSlices (默认 10) 个切片
    │   每个切片独立栅格化为占用计数图 (带抗锯齿)
    │   对每个像素: 如果在 ≥ minVotes (默认 6) 个切片中被占用 → 标记为墙壁
    │   → 生成稳定的投影图，抑制家具/噪声干扰
    │
    ▼ ⑥ rasterizePointsToCounts32F_AA()
    │   将 2D 点集栅格化为浮点占用计数图
    │   gridSize = 0.05m, resolutionScale = 2.0
    │   有效栅格分辨率 = gridSize / resolutionScale = 0.025m
    │
    ▼ ⑦ 归一化 + 保存 projection.png
    │
    ▼ ⑧ makeLineBinaryFromProjection()
    │   占用二值化 (occThreshold=5) → 中值滤波 (medianKSize=3) → 膨胀/闭运算
    │   → 形态学闭合 (closeKernelSize=7) → 可选连通域过滤
    │   → 保存 projection_lines_bin.png
    │
    ▼ ⑨ skeletonizeBinary()
    │   骨架化: 优先使用 OpenCV ximgproc 的 thinning，否则退化到 Zhang-Suen 手动实现
    │   → 保存 projection_lines_thin.png
    │
    ▼ ⑩ redrawMergedHoughLines()
    │   HoughLinesP 检测直线段
    │   → 按角度+距离+间隙合并碎片线段 (mergeAngleDeg=5°, mergeDistPx=2, mergeGapPx=12)
    │   → 将合并后的线段重绘到干净画布上
    │   → 保存 projection_lines_refined.png + projection_lines_overlay.png
    │
    ▼ ⑪ maskToPointCloud2D()
        将线段二值图转换为 2D 点云 (z=0)
        → 保存 walls_2d.pcd (线状版本) + walls_2d_fill.pcd (填充版本)
```

#### 输出文件

所有文件写入 `argv[2]` 指定的输出目录：

| 文件 | 内容 |
|------|------|
| `projection.png` | 多切片投票后的主投影图（灰度） |
| `projection_vote_counts.png` | 多切片投票计数可视化 |
| `projection_lines_bin.png` | 线段二值图（形态学处理后） |
| `projection_lines_thin.png` | 骨架化后的细线图 |
| `projection_lines_refined.png` | **Hough 线检测 + 合并后的精炼结果**（★ Step 2 的推荐输入） |
| `projection_lines_overlay.png` | 精炼线段叠加在原始投影上的可视化 |
| `walls_2d.pcd` | 2D 线状墙体点云（z=0） |
| `walls_2d_fill.pcd` | 2D 填充型墙体点云备份 |
| `slices/` | 各切片的中间投影结果 |

#### 可调参数

| 参数位置 | 名称 | 默认值 | 说明 |
|----------|------|--------|------|
| argv[3] | `gridSize` | 0.05 | 栅格基础大小（米） |
| argv[4] | `resolutionScale` | 2.0 | 分辨率放大倍数（有效分辨率 = gridSize/scale） |
| argv[5] | `lowerZ` | -1.5 | 墙壁高度下界（米，相对地面） |
| argv[6] | `upperZ` | -1.0 | 墙壁高度上界（米，相对地面） |
| argv[7] | `voxel` | 0.01 | 体素降采样大小（米） |
| argv[8] | `saveLineBinary` | 1 | 是否输出线段二值图 |
| argv[9] | `occThreshold` | 5 | 占用二值化阈值 |
| argv[10] | `medianKSize` | 3 | 中值滤波核大小 |
| argv[11] | `binDilate` | 0 | 二值图膨胀量 |
| argv[12] | `binClose` | 1 | 闭运算迭代次数 |
| argv[13] | `closeKernelSize` | 7 | 形态学闭合核大小 |
| argv[14] | `ccEnable` | 0 | 连通域过滤开关 |
| argv[15..18] | `ccMinArea/MaxArea/AspectRatio/Adaptive` | — | 连通域过滤参数 |
| argv[19] | `exportLineLikePcd` | 1 | 是否导出线状 2D 点云 |
| argv[20] | `useXimgprocThinning` | 1 | 是否使用 ximgproc 骨架化 |
| argv[21] | `houghThreshold` | 35 | Hough 累加器阈值 |
| argv[22] | `houghMinLineLength` | 25 | 最小线段长度（像素） |
| argv[23] | `houghMaxLineGap` | 6 | 最大线段间隙（像素） |
| argv[24] | `mergeAngleDeg` | 5 | 线段合并角度容差（度） |
| argv[25] | `mergeDistPx` | 2 | 线段合并距离容差（像素） |
| argv[26] | `mergeGapPx` | 12 | 线段合并间隙容差（像素） |
| argv[27] | `sharpRaster` | 0 | 是否使用锐化栅格 |
| argv[28] | `numSlices` | 10 | 多切片投票切片数 |
| argv[29] | `minVotes` | 6 | 多切片投票最小票数 |

### 5.3 Step 2：直线段检测与规则化（Python，wall_regularization_v2）

**源文件**：`l2bim/examples/python/wall_regularization_v2.py`（605 行）

#### 输入

| 参数 | 说明 |
|------|------|
| `--input` | Step 1 生成的投影图像（推荐使用 `projection_lines_refined.png`） |
| `--output-dir` | 输出目录 |

#### 处理流程

```
输入图像 (灰度/二值 PNG)
    │
    ▼ ① 读取 + 二值化 (>128 → 白)
    │   → 保存 v2_step0_raw_input.png
    │
    ▼ ② 曼哈顿方向自动对齐
    │   对所有前景像素统计方向直方图 (1° 分辨率)
    │   找到主方向后旋转图像使主墙方向水平
    │   → 保存 v2_step0_input.png, v2_step0_aligned_input.png
    │
    ▼ ③ 规则/非规则区域分块判定
    │   将图像分为 BLOCK_SIZE×BLOCK_SIZE (30×30) 的网格
    │   每个网格块: 拟合直线，如果拟合误差 < FIT_ERROR_THRESHOLD (5.0) → 规则区域
    │   否则 → 非规则区域 (家具碎片、噪声等)
    │   → 保存 v2_step1_region_mask.png (绿=规则, 红=非规则)
    │
    ▼ ④ 提取规则区域
    │   仅保留规则区域中的前景像素作为 Hough 输入
    │   → 保存 v2_step2_regular_regions.png
    │
    ▼ ⑤ Hough 线段检测
    │   对规则区域执行 HoughLinesP
    │   → 保存 v2_step3_hough_lines.png
    │
    ▼ ⑥ 共线段合并 (带路径验证)
    │   对检测到的线段:
    │   a. 按角度聚类 (水平/垂直/斜线)
    │   b. 判断是否共线 (距离+角度容差)
    │   c. 沿连接路径验证是否确实连通 (防止跨墙合并)
    │   d. 合并共线且连通的线段 (扩展端点到最远范围)
    │   → 保存 v2_step5_original_lines.png, v2_step6_extended_lines.png
    │
    ▼ ⑦ 正交拓扑交叉点修正
    │   检测近似垂直相交的线段对
    │   将端点吸附到交叉点位置，保证墙角闭合
    │   → 保存 v2_step7_combined.png (红=原始, 绿=修正后)
    │
    ▼ ⑧ 最终输出
        → 保存 v2_step8_final_result.png (★ 最终 CAD 直线段结果)
```

#### 输出文件

按处理步骤依次生成 10 个中间/最终结果图（便于调试和过程追踪）：

| 文件 | 内容 |
|------|------|
| `v2_step0_raw_input.png` | 原始输入（对齐前） |
| `v2_step0_input.png` | 二值化输入 |
| `v2_step0_aligned_input.png` | 曼哈顿对齐后的输入 |
| `v2_step1_region_mask.png` | 区域掩码（绿=规则，红=非规则） |
| `v2_step2_regular_regions.png` | 提取的规则区域 |
| `v2_step3_hough_lines.png` | Hough 线检测结果 |
| `v2_step5_original_lines.png` | 原始检测线段 |
| `v2_step6_extended_lines.png` | 共线合并后的扩展线段 |
| `v2_step7_combined.png` | 对比图（红=原始，绿=扩展） |
| `v2_step8_final_result.png` | **★ 最终 CAD 直线段结果** |

> 另有变体版本 `wall_regularization_v2_fix.py`（619 行），跳过区域分类步骤（直接使用所有前景）并停止共线合并的增长，适用于碎片化较少的场景。

### 5.4 编译与运行

#### macOS / Linux 编译

```bash
cd l2bim
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make pcd_projection
```

#### Windows 编译

```powershell
cd l2bim
cmake -S . -B build_vs17_x64 -G "Visual Studio 17 2022" -A x64
cmake --build build_vs17_x64 --config Release --target pcd_projection
```

#### 运行 Step 1（C++ 投影）

```bash
# macOS / Linux
cd l2bim
./bin/pcd_projection <input.pcd> <output_dir> [可选参数...]

# 示例: 使用默认参数
./bin/pcd_projection pcds/001.pcd output/001

# 示例: 自定义参数
./bin/pcd_projection pcds/001.pcd output/001 0.05 2.0 -1.5 -1.0 0.01
```

```powershell
# Windows
.\build_vs17_x64\Release\pcd_projection.exe .\pcds\001.pcd .\output\001
```

#### 运行 Step 2（Python 规则化）

```bash
cd l2bim
python examples/python/wall_regularization_v2.py \
    --input output/001/projection_lines_refined.png \
    --output-dir output/001
```

> 脚本末尾会弹出可视化窗口（`cv2.imshow`），关闭窗口后进程结束。

---

## 6. 完整数据流转图

以下是三个阶段之间所有文件的流转关系汇总：

```
═══════════════════════════════════════════════════════════════════════
              文件级数据流转 (File-level Data Flow)
═══════════════════════════════════════════════════════════════════════

[采集设备 (MetaCam Air)]
    │
    ├──→ color/left/*.jpg          ──┐
    ├──→ color/right/*.jpg          │
    ├──→ colorized.pcd             ──┤──→ 阶段二输入目录
    ├──→ transforms.json           ──┤
    └──→ config.yaml               ──┘
              │
              │ (RGB 图像: left/ + right/)
              ▼
═══════════ 阶段一 (rma-dino-sam) ════════════════════════════════════

    color/left/1765952490650809088.jpg
    color/right/1765952490650811904.jpg
              │
              ▼
    prediction_no_augment/
    ├── 1765952490650809088_mask.png      ← 16-bit 实例掩码 (left)
    ├── 1765952490650809088_label.json    ← {raw_tags, tags, mask[{value, labels, box}]}
    ├── 1765952490650811904_mask.png      ← 16-bit 实例掩码 (right)
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
              │  removed.pcd (结构背景点云)
              ▼
═══════════ 阶段三 Step 1: pcd_projection (C++) ═══════════════════

    输入: removed.pcd (或其他 .pcd/.ply)
              │
              ▼
    output/
    ├── projection.png
    ├── projection_lines_refined.png  ★
    ├── projection_lines_*.png
    ├── walls_2d.pcd
    ├── walls_2d_fill.pcd
    └── slices/
              │
              ▼
═══════════ 阶段三 Step 2: wall_regularization_v2 (Python) ════════

    输入: projection_lines_refined.png
              │
              ▼
    output/
    ├── v2_step0_*.png (中间结果)
    ├── v2_step1~7_*.png (过程图)
    └── v2_step8_final_result.png  ★
              │
              ▼
═══════════ 最终结果 ═════════════════════════════════════════════════

    v2_step8_final_result.png → 规则化 2D CAD 直线段平面图
    walls_2d.pcd              → 2D 线状墙体点云
    projection.png            → 投影参考图
```

---

## 7. 目录结构与文件索引

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
│   │   ├── mapping/
│   │   │   ├── SemanticMapping.cpp    # 核心: 投影融合 + 实例管理
│   │   │   ├── SemanticDict.h         # 语义类别定义 (floor/ceiling/carpet)
│   │   │   └── Instance.h             # 实例数据结构
│   │   └── tools/
│   │       ├── IO.cpp                 # 数据加载 (含 construct_sorted_frame_table_3_2)
│   │       └── IO.h                   # IO 函数声明
│   ├── cmake/                        # CMake 依赖配置
│   └── test_data/                    # 示例数据
│       ├── colorized.pcd             # SLAM 点云
│       ├── config.yaml               # 配置文件
│       ├── transforms.json           # 位姿信息
│       ├── color/left/*.jpg          # 左相机图像
│       ├── color/right/*.jpg         # 右相机图像
│       └── prediction_no_augment/    # 阶段一输出
│
├── l2bim/                            # ===== 阶段三 =====
│   ├── CMakeLists.txt                # 构建配置 (目标: pcd_projection 等)
│   ├── README.md                     # l2bim 项目说明
│   ├── examples/
│   │   ├── cpp/
│   │   │   ├── pcd_projection.cpp    # ★ Step 1: 点云投影主程序 (1133行)
│   │   │   ├── pcd2line_test.cpp     # 替代方案: 投影+ED线检测完整pipeline
│   │   │   ├── projection_2d.cpp     # 替代方案: 密度百分位投影
│   │   │   └── img_plus.cpp          # 图像后处理工具 (形态学+连通域)
│   │   └── python/
│   │       ├── wall_regularization_v2.py     # ★ Step 2: 直线段检测与规则化 (605行)
│   │       └── wall_regularization_v2_fix.py # 变体: 跳过区域分类 (619行)
│   ├── src/
│   │   ├── frontend/geometry/
│   │   │   ├── point_cloud.cpp       # PCD2d 类: 地面拟合+坐标变换+投影
│   │   │   ├── lineseg.cpp           # 线段管理
│   │   │   ├── corner.cpp            # 角点检测
│   │   │   ├── ED.cpp / EDLines.cpp  # ED 线段检测器
│   │   │   └── NFA.cpp               # NFA 验证
│   │   └── utils/
│   │       └── raster_utils.cpp      # PointRaster/LineRaster 栅格化工具
│   ├── pcds/                         # 测试点云 (001~008.pcd, 3.3.pcd)
│   ├── output/                       # 输出结果
│   │   ├── detect/                   # 线检测结果
│   │   └── projection/              # 投影结果
│   └── Thirdparty/
│       ├── point-sam/                # 点云采样库
│       └── nanoflann/                # 快速最近邻搜索
│
├── PIPELINE.md                       # ← 本文档
├── INSTALL.md                        # 环境部署指南
└── README.md                         # 项目简介
```

---

## 8. 快速上手

> 环境搭建、依赖安装与版本管理请参考 **[INSTALL.md](INSTALL.md)**。

### 一步步运行

```bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: 阶段一 — 2D 语义分割 (GPU 服务器)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cd rma-dino-sam
export WEIGHTS_DIR="/path/to/weights"

# 必须分别处理左右相机图像！
python main.py "/path/to/data/color/left"  "/path/to/data/prediction_no_augment"
python main.py "/path/to/data/color/right" "/path/to/data/prediction_no_augment"

# 输出: prediction_no_augment/ 目录 (包含 left + right 的 mask + label)
# 该目录需与 color/ 同级，放在阶段二数据目录下


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: 阶段二 — 3D 语义实例融合
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cd PointCloud_Segement_v0/build
cmake .. && make
./Test_main ../test_data ../test_data/outputs/run01

# 输出:
#   ├── segement.pcd     (实例着色点云)
#   ├── removed.pcd      (结构背景点云 → 阶段三输入)
#   ├── info.txt         (语义标签)
#   ├── points.txt       (点索引, 二进制)
#   └── config.txt       (配置备份)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: 阶段三 Step 1 — 点云投影 (C++, l2bim)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cd l2bim

# 编译 (首次)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make pcd_projection
cd ..

# 运行 (输入为阶段二的 removed.pcd)
./bin/pcd_projection \
    ../PointCloud_Segement_v0/test_data/outputs/run01/removed.pcd \
    output/run01

# 输出:
#   ├── projection.png                   (投影主图)
#   ├── projection_lines_refined.png     (★ 精炼线段图 → Step 2 输入)
#   ├── projection_lines_*.png           (中间结果)
#   ├── walls_2d.pcd                     (线状 2D 点云)
#   ├── walls_2d_fill.pcd                (填充 2D 点云)
#   └── slices/                          (切片中间结果)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: 阶段三 Step 2 — 直线段检测与规则化 (Python, l2bim)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

python examples/python/wall_regularization_v2.py \
    --input output/run01/projection_lines_refined.png \
    --output-dir output/run01

# 输出:
#   ├── v2_step0_raw_input.png           (对齐前)
#   ├── v2_step0_input.png               (二值化)
#   ├── v2_step0_aligned_input.png       (曼哈顿对齐)
#   ├── v2_step1_region_mask.png         (区域掩码)
#   ├── v2_step2_regular_regions.png     (规则区域)
#   ├── v2_step3_hough_lines.png         (Hough 检测)
#   ├── v2_step5_original_lines.png      (原始线段)
#   ├── v2_step6_extended_lines.png      (合并扩展)
#   ├── v2_step7_combined.png            (对比图)
#   └── v2_step8_final_result.png        (★ 最终 CAD 直线段结果)
```

### 可调参数速查

| 参数 | 默认值 | 在哪里改 | 影响 |
|------|--------|---------|------|
| 相机内参 (fx, fy, cx, cy) | — | `config.yaml` | 2D→3D 投影精度 |
| `voxel_length` | 0.02 | `config.yaml` | 体素分辨率，越小越精细 |
| `min_det_masks` | 2000 | `config.yaml` | 最小检测像素数，过滤小实例 |
| `merge_iou` | 0.3 | `config.yaml` | 实例合并 IoU 阈值 |
| `gridSize` | 0.05 | `pcd_projection` argv[3] | 投影栅格大小（米） |
| `resolutionScale` | 2.0 | `pcd_projection` argv[4] | 分辨率放大倍数 |
| `lowerZ` / `upperZ` | -1.5 / -1.0 | `pcd_projection` argv[5-6] | 墙壁高度范围（米） |
| `numSlices` | 10 | `pcd_projection` argv[28] | 多切片投票数 |
| `minVotes` | 6 | `pcd_projection` argv[29] | 投票最小阈值 |
| `houghThreshold` | 35 | `pcd_projection` argv[21] | Hough 累加器阈值 |
| `houghMinLineLength` | 25 | `pcd_projection` argv[22] | 最小线段长度（像素） |
| `mergeAngleDeg` | 5 | `pcd_projection` argv[24] | 线段合并角度容差 |
| `BLOCK_SIZE` | 30 | `wall_regularization_v2.py` 源码 | 区域分类网格大小 |
| `FIT_ERROR_THRESHOLD` | 5.0 | `wall_regularization_v2.py` 源码 | 规则区域判定阈值 |
