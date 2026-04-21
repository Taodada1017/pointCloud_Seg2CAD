# PointCloud2CAD

> **基于开放词汇语义分割的室内三维点云语义实例分割与 CAD 平面图生成**

本项目实现了一套从 LiDAR 扫描点云到 2D CAD 平面图的自动化处理管线 (Pipeline)，核心思路是利用 2D 基础视觉模型（RAM + GroundingDINO + SAM）对多视角图像进行开放词汇语义分割，再通过自研的多帧 2D→3D 投影融合算法将语义标签迁移至三维点云，最终提取建筑结构生成 CAD 直线段平面图。

---

## 项目结构

```
PointCloud2CAD/
├── rma-dino-sam/               # 阶段一：2D 开放词汇语义分割 (Python)
│   ├── main.py                 # 主入口脚本
│   ├── recognize-anything/     # RAM 图像标签识别
│   ├── GroundingDINO/          # Grounding DINO 开放集目标检测
│   ├── segment_anything/       # SAM 分割一切模型
│   └── clip/                   # OpenAI CLIP 模型
│
├── PointCloud_Segement_v0/     # 阶段二：3D 语义实例融合 (C++)
│   ├── main.cpp                # 主入口
│   ├── src/                    # 核心源码
│   │   ├── mapping/            # 语义建图：体素哈希、贝叶斯标签融合、实例管理
│   │   └── tools/              # IO、计时、可视化工具
│   ├── cmake/                  # CMake 依赖配置
│   └── test_data/              # 示例数据（含完整测试集）
│
├── l2bim/                      # 阶段三：点云 → 2D CAD 直线段提取 (C++/Python)
│   ├── examples/
│   │   ├── cpp/pcd_projection.cpp          # ★ Step 1: 点云投影 (C++)
│   │   └── python/wall_regularization_v2.py # ★ Step 2: 直线段检测与规则化 (Python)
│   ├── src/                    # C++ 源码（投影、栅格化、线段检测）
│   ├── pcds/                   # 测试点云
│   └── Thirdparty/             # 第三方库 (nanoflann, point-sam)
│
├── PIPELINE.md                 # 详细流程文档
├── INSTALL.md                  # 环境部署指南
└── README.md                   # ← 本文档
```

---

## Pipeline 总览

```
RGB 图像序列                          LiDAR 点云 (.pcd)
      │                                      │
      ▼                                      │
┌─────────────┐                              │
│  RAM 标签   │   提取图像语义标签             │
└─────┬───────┘                              │
      ▼                                      │
┌─────────────┐                              │
│ GroundingDINO│   开放词汇目标检测           │
└─────┬───────┘                              │
      ▼                                      │
┌─────────────┐                              │
│    SAM      │   实例级分割 mask             │
└─────┬───────┘                              │
      │                                      │
      ▼                                      ▼
┌──────────────────────────────────────────────┐
│         2D → 3D 投影 & 多帧语义融合          │  ← 核心贡献
│  (体素哈希建图 + 贝叶斯标签 + 实例合并)      │
└──────────────────┬───────────────────────────┘
                   ▼
          3D 语义实例点云
                   │
                   ▼
┌──────────────────────────────┐
│  点云投影 → 直线段检测 → CAD  │
└──────────────────────────────┘
```

| 阶段 | 模块 | 语言 | 核心功能 |
|------|------|------|---------|
| **一** | `rma-dino-sam/` | Python | RAM + GroundingDINO + SAM 开放词汇语义实例分割 |
| **二** | `PointCloud_Segement_v0/` | C++ | 2D→3D 反投影、多帧贝叶斯融合、跨帧实例合并 |
| **三** | `l2bim/` | C++ + Python | RANSAC 分离 → 多切片投影 → 骨架化 → Hough 检测 → 规则化 |

---

## 数据准备

阶段二所需的输入数据目录结构：

```
data_folder/
├── color/
│   ├── left/         # 左相机去畸变 RGB 图像 (.jpg)
│   └── right/        # 右相机去畸变 RGB 图像 (.jpg)
├── colorized.pcd     # LiDAR 点云 (PCD 格式)
├── config.yaml       # 相机内参配置 (fx, fy, cx, cy, w, h)
├── transforms.json   # 图像位姿信息 (由 SLAM 导出)
└── prediction_no_augment/
    ├── <timestamp>_mask.png     # ← 阶段一输出
    └── <timestamp>_label.json   # ← 阶段一输出
```

> `test_data/` 中已提供完整的示例数据集，可直接运行 demo。

---

## 快速运行

> 环境搭建详见 **[INSTALL.md](INSTALL.md)**。

```bash
# ━━━━ 阶段一：2D 语义分割 (GPU 推荐) ━━━━
cd rma-dino-sam
export WEIGHTS_DIR="/path/to/weights"
# 必须分别处理左右相机图像，输出到同一目录
python main.py "<数据目录>/color/left"  "<数据目录>/prediction_no_augment"
python main.py "<数据目录>/color/right" "<数据目录>/prediction_no_augment"

# ━━━━ 阶段二：3D 语义实例融合 ━━━━
cd PointCloud_Segement_v0/build
cmake .. && make
./Test_main ../test_data ../test_data/outputs/run01

# ━━━━ 阶段三 Step 1：点云投影 (C++) ━━━━
cd l2bim
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make pcd_projection && cd ..
./bin/pcd_projection ../PointCloud_Segement_v0/test_data/outputs/run01/removed.pcd output/run01

# ━━━━ 阶段三 Step 2：直线段检测与规则化 (Python) ━━━━
python examples/python/wall_regularization_v2.py \
    --input output/run01/projection_lines_refined.png \
    --output-dir output/run01
```

---

## 主要贡献

本项目中以下模块为自主开发：

- **体素哈希建图 (VoxelHashMap)** — 高效的 3D 空间索引结构
- **2D → 3D 语义投影** — 利用相机内外参将 2D 分割结果反投影到点云
- **贝叶斯标签融合** — 基于多帧观测的概率语义标签决策
- **多帧数据关联与实例合并** — 跨帧实例 ID 一致性维护
- **点云投影与 CAD 直线段提取** — 多切片投票投影 + 骨架化 + Hough 检测 + 曼哈顿规则化

复用的开源模型：RAM、GroundingDINO、SAM、CLIP。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| **[PIPELINE.md](PIPELINE.md)** | 三个阶段的完整串联流程、数据格式与流转细节 |
| **[INSTALL.md](INSTALL.md)** | 环境部署、依赖安装、版本管理与冲突解决 |
| **[l2bim/README.md](l2bim/README.md)** | 阶段三模块的详细说明 |

---

## 许可证

各子模块遵循其各自的开源许可证（详见子目录中的 LICENSE 文件）。
