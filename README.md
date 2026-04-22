# PointCloud2CAD

> **面向室内BIM重建的点云语义分割与CAD平面图自动生成**

给定室内场景的 LiDAR 彩色点云 + 多视角图像 + 相机位姿，自动输出带真实尺度的 2D CAD 平面图。

```
输入:                                    输出:
├── colorized.pcd (3500万 3D彩色点)       ├── segement.pcd  (实例分割点云)
├── RGB/left/*.jpg  (70张左相机)          ├── removed.pcd   (纯结构点云)
├── RGB/right/*.jpg (75张右相机)          ├── floorplan.dxf  (AutoCAD 格式)
├── transforms.json (150帧位姿+内参)      └── floorplan.svg  (矢量图, 含比例尺)
└── config.yaml     (算法参数)
```

---

## 三阶段技术路线

```
阶段一（逐张图像处理）               阶段二（逐帧融合到3D）              阶段三（3D → 2D CAD）
"图里有什么？轮廓在哪？"             "对应3D空间的哪些点？"              "墙在哪里？画出线段"
      │                                    │                                  │
      ▼                                    ▼                                  ▼
 RAM + DINO + SAM                    2D→3D反投影                      RANSAC地板分离
 开放词汇语义分割                     3D点集IoU关联                    自适应高度检测
      │                              贝叶斯标签融合                    多层切片投票
      ▼                              实例合并+家具移除                  骨架化+Hough检测
 145对 mask+label                         │                           共线合并→CAD输出
                                          ▼                                  │
                                   语义实例点云 + 纯结构点云                  ▼
                                                                     79条墙线 (DXF/SVG)
```

| 阶段 | 模块 | 语言 | 核心功能 |
|------|------|------|---------|
| **一** | `rma-dino-sam/` | Python | RAM 标签识别 → GroundingDINO 开放集检测 → SAM 实例分割 |
| **二** | `PointCloud_Segement_v0/` | C++ | 体素哈希加速视锥体筛选 → 2D→3D反投影 → 3D点集IoU关联 → 贝叶斯融合 |
| **三** | `l2bim/` | C++ + Python | Z直方图自适应高度 → 多切片投票投影 → 骨架化+Hough → 共线合并 → CAD |

---

## 核心创新点

### 🔵 基于3D点集级IoU的语义实例数据关联（阶段二）

传统方法在2D图像平面计算IoU进行数据关联，受视角变化影响严重——同一物体从不同角度拍摄时，2D mask可能完全不重叠。

本方法将关联提升到3D空间：

```
传统方法：  IoU_2D = |mask_A ∩ mask_B| / |mask_A ∪ mask_B|     （像素级，视角敏感）
本文方法：  IoU_3D = |points_A ∩ points_B| / |points_A ∪ points_B|  （点索引级，视角鲁棒）
```

- **体素哈希加速**：DDA算法遍历视锥体穿过的体素，避免暴力遍历3500万点，加速约40倍
- **双向最大匹配**：行/列方向一致才确认关联，歧义对用3D IoU决定是否合并
- **贝叶斯多帧融合**：似然矩阵将开放词汇标签映射到预定义类别，多帧累积后验概率

### 🟢 自适应点云投影参数估计（阶段三）

对结构点云Z轴建立直方图，自动检测地板/天花板高度，替代硬编码参数：

| 参数方案 | 投影有效像素 | 检测墙线数 |
|---------|------------|-----------|
| 硬编码 [-1.5, -1.0] | 3,625 | 5 条 |
| **自适应检测** | **55,000+** | **79 条** |

### 🟢 面向室内建筑的语义先验过滤（阶段一）

利用CLIP计算RAM输出标签与预定义建筑类别的语义相似度，过滤无关标签（"tape"、"hallway"等），减少误检。

### 🟢 基于视角覆盖的关键帧选取（阶段一）

根据相机位姿变化量选取关键帧（145帧→~30帧），预期降低80%计算量。

---

## 实测数据

在真实大型室内场景（69m × 94m × 20m）上的端到端结果：

| 指标 | 数值 |
|------|------|
| 原始点云 | 34,975,020 点 |
| 检测实例数 | 278 个（floor 47、wall 21、ceiling 5、chair 20 ...） |
| 移除非结构点 | 9,292,704 点（26.6%） |
| 保留结构点 | 25,682,317 点 |
| 检测墙线 | 79 条，总长 241.0m |
| 场景覆盖 | 59.7m × 90.2m |
| 分辨率 | 2.5 cm/像素 |

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
│   │   ├── mapping/            # 体素哈希建图、贝叶斯标签融合、实例管理
│   │   └── tools/              # IO、计时、可视化工具
│   └── cmake/                  # CMake 依赖配置
│
├── l2bim/                      # 阶段三：点云 → 2D CAD 直线段提取 (C++/Python)
│   ├── examples/
│   │   ├── cpp/pcd_projection.cpp              # Step 1: 点云投影 (C++)
│   │   └── python/wall_regularization_v3.py    # Step 2: 直线段检测与规则化 (Python)
│   ├── src/                    # C++ 源码（投影、栅格化、线段检测）
│   └── Thirdparty/             # 第三方库 (nanoflann, point-sam)
│
├── docs/                       # 技术文档
│   ├── 三阶段技术路线详细介绍.md    # 含真实数据和数学推导的详细技术文档
│   └── 毕设技术路线介绍.md         # 研究背景、创新点与实验设计
│
├── PIPELINE.md                 # 三阶段完整串联流程与数据流转
├── environment.md              # 环境部署指南
└── README.md                   # ← 本文档
```

---

## 快速运行

> 环境搭建详见 **[environment.md](environment.md)**。

### 阶段一：2D 语义分割

```bash
cd rma-dino-sam
export WEIGHTS_DIR="/path/to/weights"

# 分别处理左右相机图像，输出到同一目录
python main.py "<数据目录>/color/left"  "<数据目录>/prediction_no_augment"
python main.py "<数据目录>/color/right" "<数据目录>/prediction_no_augment"
```

### 阶段二：3D 语义实例融合

```bash
cd PointCloud_Segement_v0/build
cmake .. && make -j$(nproc)
./Test_main ../test_data ../test_data/outputs/run01
```

### 阶段三：CAD 线段提取

```bash
# Step 1: 点云投影 (C++)
cd l2bim
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make pcd_projection && cd ..
./bin/pcd_projection ../PointCloud_Segement_v0/test_data/outputs/run01/removed.pcd output/run01

# Step 2: 直线段检测与规则化 (Python)
python examples/python/wall_regularization_v3.py \
    --input output/run01/projection_lines_refined.png \
    --output-dir output/run01
```

---

## 数据准备

```
data_folder/
├── color/
│   ├── left/         # 左相机去畸变 RGB 图像 (.jpg)
│   └── right/        # 右相机去畸变 RGB 图像 (.jpg)
├── colorized.pcd     # LiDAR 点云 (PCD 格式)
├── config.yaml       # 相机内参 (fx, fy, cx, cy, w, h)
├── transforms.json   # 图像位姿 (SLAM 导出)
└── prediction_no_augment/
    ├── <timestamp>_mask.png     # ← 阶段一输出
    └── <timestamp>_label.json   # ← 阶段一输出
```

---

## 文档索引

| 文档 | 内容 |
|------|------|
| **[PIPELINE.md](PIPELINE.md)** | 三阶段完整串联流程、数据格式与流转细节 |
| **[environment.md](environment.md)** | 环境部署、依赖安装、版本冲突解决 |
| **[docs/三阶段技术路线详细介绍.md](docs/三阶段技术路线详细介绍.md)** | 含真实数据和数学推导的详细技术文档 |
| **[docs/毕设技术路线介绍.md](docs/毕设技术路线介绍.md)** | 研究背景、创新点与实验设计 |
| **[l2bim/README.md](l2bim/README.md)** | 阶段三模块详细说明 |

---

## 当前进展

| 模块 | 状态 | 说明 |
|------|------|------|
| 阶段一 基础管线 | ✅ 已完成 | RAM+DINO+SAM 级联，145张图全部处理 |
| 阶段一 语义过滤 | 🔲 待实现 | CLIP语义相似度过滤 |
| 阶段一 关键帧选取 | 🔲 待实现 | 视角覆盖策略 |
| 阶段二 核心融合 | ✅ 已完成 | 3D点集IoU关联 + 实例管理 |
| 阶段二 贝叶斯融合 | 🔲 待完善 | 代码框架已有，需设计似然矩阵 |
| 阶段三 投影+线段 | ✅ 已完成 | 自适应高度 + 多层投票 + DXF输出 |
| 多场景验证 | 🔲 待进行 | 需3-5个不同场景数据 |

---

## 许可证

各子模块遵循其各自的开源许可证（详见子目录中的 LICENSE 文件）。
