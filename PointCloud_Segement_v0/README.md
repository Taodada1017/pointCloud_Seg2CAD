# PointCloud_Segement_v0 — 3D 语义实例融合

> **阶段二**：将 2D 语义分割结果投影到 3D 点云，通过多帧融合实现语义实例级点云分割。

---

## 功能概述

本模块是 PointCloud2CAD Pipeline 的**第二阶段**，也是本项目的**核心贡献模块**。负责将阶段一输出的 2D 分割结果（mask + label）与 LiDAR 点云进行融合，生成 3D 语义实例分割点云。

```
阶段一输出 (mask + label)
         │
         ▼
   ┌───────────────┐     ┌─────────────┐
   │ 2D → 3D 投影  │────►│ 体素哈希建图 │
   │（相机内外参）    │     │(VoxelHashMap)│
   └───────────────┘     └──────┬──────┘
                                │
                    ┌───────────▼───────────┐
                    │ 贝叶斯标签融合 + 数据关联 │
                    │ + 实例合并 (IoU + 语义)   │
                    └───────────┬───────────┘
                                │
                    3D 语义实例分割点云
```

### 自主开发模块

- **体素哈希建图 (VoxelHashMap)** — 高效的 3D 空间索引，加速点云查询
- **2D → 3D 语义投影** — 利用相机内外参将像素级分割结果反投影到点云
- **贝叶斯标签融合 (BayesianLabel)** — 基于多帧观测的概率语义标签决策
- **多帧数据关联与实例合并** — 跨帧实例 ID 一致性维护，IoU + 语义相似度判断
- **地板/天花板/重叠实例的智能合并** — 处理结构性语义的特殊合并策略

---

## 目录结构

```
PointCloud_Segement_v0/
├── main.cpp                    # ★ 主入口
├── CMakeLists.txt              # CMake 构建配置
├── README.md                   # 本文件
├── cmake/                      # 外部依赖 CMake 配置
│   ├── eigen.cmake
│   ├── jsoncpp.cmake
│   ├── open3d.cmake
│   ├── opencv.cmake
│   └── tbb.cmake
├── src/                        # 核心源码
│   ├── Common.h                # 全局配置结构体 (InstanceConfig, MappingConfig 等)
│   ├── CMakeLists.txt          # 源码库构建
│   ├── mapping/                # ★ 语义建图核心
│   │   ├── SemanticMapping.cpp/h   # 语义建图主类（投影、融合、合并、导出）
│   │   ├── VoxelHashMap.cpp/h      # 体素哈希空间索引
│   │   ├── BayesianLabel.h         # 贝叶斯标签融合
│   │   ├── Instance.cpp/h          # 实例对象管理
│   │   ├── Detection.cpp/h         # 检测结果封装
│   │   ├── SubVolume.cpp/h         # 子体积管理
│   │   └── SemanticDict.h          # 语义字典
│   ├── cluster/                # 图优化
│   │   └── PoseGraph.cpp/h
│   ├── sgloop/                 # 场景图
│   │   └── Graph.cpp/h
│   └── tools/                  # 工具类
│       ├── IO.cpp/h            # 数据读写（位姿、标签、点云等）
│       ├── Utility.cpp/h       # 通用工具函数
│       ├── TicToc.h            # 计时工具
│       ├── Color.h             # 颜色编码
│       └── Tools.h             # 杂项
└── test_data/                  # 示例数据（结构演示）
    ├── color/                  # 去畸变 RGB 图像
    │   ├── left/               # 左相机图像 (.jpg)
    │   └── right/              # 右相机图像 (.jpg)
    ├── colorized.pcd           # LiDAR 点云（~400MB，不入 git）
    ├── config.yaml             # 相机内参配置
    ├── transforms.json         # 图像位姿信息（由 Studio 导出）
    └── prediction_no_augment/  # 阶段一输出的分割结果
        ├── {timestamp}_label.json
        └── {timestamp}_mask.png
```

---

## 环境依赖

> 详细的跨设备安装流程请参阅项目根目录下的 [`INSTALL.md`](../INSTALL.md)。

| 库 | 参考版本 | 说明 |
|----|---------|------|
| **Eigen** | 3.3.7 | 线性代数 |
| **Jsoncpp** | 1.7.4 | JSON 解析 |
| **OpenCV** | 4.2.0 | 图像处理 |
| **Open3D** | 0.19 | 点云处理与可视化 |
| **TBB** | 2020.1 | 并行计算 |
| **CMake** | ≥ 4.0 | 构建系统 |
| **Make** | ≥ 4.2 | 编译 |

### 编译

```bash
cd PointCloud_Segement_v0
mkdir build && cd build
cmake ..
make -j$(nproc)
```

编译成功后会在 `build/` 目录下生成 `Test_main` 可执行文件。

---

## 输入数据格式

数据文件夹需要保持以下结构（**路径命名需完全一致**）：

```
/path/to/data/
├── color/
│   ├── left/                   # 左相机去畸变 RGB 图像 (.jpg)
│   └── right/                  # 右相机去畸变 RGB 图像 (.jpg)
├── colorized.pcd               # LiDAR 点云（PCD 格式）
├── config.yaml                 # 相机内参配置
├── transforms.json             # 图像位姿（由 Studio 导出）
└── prediction_no_augment/      # 阶段一输出
    ├── {timestamp}_label.json  # 语义标签
    └── {timestamp}_mask.png    # 分割掩码（16 位 PNG）
```

### 各文件说明

| 文件/目录 | 说明 |
|-----------|------|
| `color/left/` `color/right/` | 去畸变后的双目 RGB 图像，由 Studio 的 undistort 功能输出 |
| `transforms.json` | Studio 导出的图像位姿信息 |
| `config.yaml` | 相机内参配置（`fx`, `fy`, `cx`, `cy`, `w`, `h`），设备不同需修改 |
| `prediction_no_augment/` | 阶段一的输出结果（`_mask.png` + `_label.json`） |
| `colorized.pcd` | 点云文件，需使用 PCD 格式。可用 CloudCompare 将 `.las` 转为 `.pcd` |

> 💡 `test_data/` 中提供了示例数据结构。大文件（如 `colorized.pcd`）请通过网盘分发。

---

## 运行

```bash
cd PointCloud_Segement_v0/build
./Test_main "<数据文件夹路径>" "<输出文件夹路径>"
```

如不传参数，将使用 `main.cpp` 中的默认路径。

---

## 输出文件说明

```
<输出文件夹>/
├── segement.pcd     # 分割后的实例点云
├── removed.pcd      # 移除已识别物体后的点云
├── info.txt         # 实例 → 语义标签映射矩阵
├── points.txt       # 每个实例的点索引（⚠️ 二进制格式，不要用文本编辑器打开！）
└── config.txt       # 运行配置信息
```

| 文件 | 说明 |
|------|------|
| `segement.pcd` | 分割后的地图点云。不同颜色代表不同实例 ID（颜色由 ID 哈希编码生成） |
| `removed.pcd` | 从原始 SLAM 地图中剔除所有已识别物体后的效果 |
| `info.txt` | 各实例物体对应的语义标签及其贝叶斯融合后的概率分布 |
| `points.txt` | 每个实例物体的点云索引数组（二进制存储，用于后续阶段读取） |
| `config.txt` | 本次运行的完整配置参数 |

---

## 配置参数说明

配置文件 `config.yaml` 中的关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `voxel_length` | 0.02 | 体素大小（米） |
| `depth_max` | 4.0 | 最大投影深度（米） |
| `min_det_masks` | 2000 | 最小检测掩码像素数 |
| `min_iou` | 0.3 | 最小 IoU 阈值（数据关联） |
| `merge_iou` | 0.3 | 实例合并 IoU 阈值 |
| `merge_inflation` | 4.0 | 合并膨胀系数 |
| `search_radius` | 4.0 | 活跃实例搜索半径（米） |
| `shape_min_points` | 500 | 导出实例的最小点数 |
| `update_period` | 20 | 实例更新周期（帧） |

---

## 注意事项

- 内部数据文件夹路径采用**硬编码命名**（如 `color/left/`、`prediction_no_augment/`），请保持一致
- `colorized.pcd` 必须为 PCD 格式。如原始数据为 `.las`，请事先在 CloudCompare 中转换
- 如果只想验证功能，可以使用设备采集的 `Preview.pcd`，改名为 `colorized.pcd` 即可
