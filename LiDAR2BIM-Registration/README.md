# LiDAR2BIM-Registration — 点云配准与 CAD 生成

> **阶段三**：将语义实例分割后的点云与 BIM 模型进行配准，提取建筑结构生成 CAD 平面图。

---

## 功能概述

本模块是 PointCloud2CAD Pipeline 的**第三阶段**，负责将阶段二输出的语义实例点云与 BIM（建筑信息模型）进行全局配准，并从点云中提取墙面线段，最终生成 2D CAD 平面图。

```
3D 语义实例点云
       │
       ▼
┌────────────────┐     ┌──────────────┐
│  点云预处理     │────►│  线段提取     │
│ (pc2img 投影)  │     │  (lineseg)   │
└────────────────┘     └──────┬───────┘
                              │
                    ┌─────────▼─────────┐
                    │ 几何特征描述子构建   │
                    │ (TriDesc 三角描述子) │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Hough 投票配准      │
                    │ + 验证与评分        │
                    └─────────┬─────────┘
                              │
                    CAD 平面图输出
```

### 核心算法

- **点云 → 2D 投影 (pc2img)** — 将 3D 点云投影到 2D 平面图
- **线段提取 (lineseg)** — 从 2D 投影中提取墙面线段
- **三角几何描述子 (TriDesc)** — 基于角点三元组的几何特征
- **Hough 投票全局配准** — 在 SE(2) 空间中搜索最优位姿
- **验证与评分 (Verification)** — 基于栅格化的配准质量评估

---

## 目录结构

```
LiDAR2BIM-Registration/
├── CMakeLists.txt              # CMake 构建配置
├── requirements.txt            # Python 预处理依赖
├── README.md                   # 本文件
├── src/                        # C++ 核心算法源码
│   ├── frontend/
│   │   ├── geometry/           # 几何处理 (BIM, 子图, 线段, 点云, 角点)
│   │   └── feature/            # 特征描述子 + 匹配
│   ├── backend/
│   │   ├── reglib.cpp          # 全局配准入口
│   │   └── hough/              # Hough 投票 + 位姿验证
│   └── utils/                  # 工具 (配置、IO、评估、栅格化)
├── include/                    # C++ 头文件（与 src/ 对应）
│   ├── frontend/
│   ├── backend/
│   ├── utils/
│   └── global_definition/
├── preprocess/                 # Python 预处理脚本
│   ├── config.py               # 配置管理
│   ├── utils.py                # 通用工具
│   ├── clustering_utils.py     # 并查集工具
│   ├── geometry/               # 几何处理
│   │   ├── pc2img.py           # 点云 → 2D 投影
│   │   ├── lineseg.py          # 线段提取
│   │   ├── bim.py              # BIM 处理
│   │   ├── corner.py           # 角点检测
│   │   ├── point_cloud.py      # 点云处理
│   │   └── submap.py           # 子图管理
│   └── datasets/               # 数据集管理
│       ├── dir.py              # 目录工具
│       ├── libim_ust_utils.py  # LiBIM-UST 数据集工具
│       ├── lineseg_manager.py  # 线段管理器
│       ├── merge_pcds.py       # 点云合并
│       ├── pcd2d_manager.py    # 2D 点云管理
│       ├── station_manager.py  # 站点管理
│       └── submap3d_manager.py # 3D 子图管理
├── examples/                   # 示例代码
│   ├── cpp/
│   │   ├── demo_reg.cpp        # 单次配准 demo
│   │   └── reg_bm.cpp          # 批量基准测试
│   ├── python/data_process/
│   │   ├── submap3d_generator.py   # 子图生成
│   │   └── make_benchmarks.py      # 生成基准数据
│   └── scripts/
│       └── benchmark.sh        # 批量评估脚本
├── configs/                    # 场景配置文件
│   ├── interval/               # 按间隔划分（15m/30m）
│   └── pointsam/               # PointSAM 配置
├── Thirdparty/                 # 第三方库
│   ├── backward-cpp/           # 调试回溯
│   ├── nanoflann/              # KD-Tree 快速近邻搜索
│   └── point-sam/              # 平面分割
└── docs/                       # 文档
    ├── install.md              # 安装指南
    ├── prepare.md              # 数据准备指南
    ├── demo.md                 # Demo 运行说明
    └── benchmark.md            # 基准测试说明
```

---

## 环境配置

> 详细的跨设备安装流程请参阅项目根目录下的 [`INSTALL.md`](../INSTALL.md)。

### C++ 依赖

| 库 | 说明 |
|----|------|
| **Boost** | 通用 C++ 库 |
| **Eigen** | 线性代数 |
| **yaml-cpp** | YAML 配置解析 |
| **glog** | 日志 |
| **PCL** | 点云处理 |
| **OpenMP** | 并行加速 |
| **OpenCV** | 图像处理 |
| **CGAL**（可选） | 计算几何 |

```bash
# Ubuntu 安装系统依赖
sudo apt-get update
sudo apt install libboost-dev libyaml-cpp-dev libomp-dev
sudo apt-get install libgmp-dev libmpfr-dev
sudo apt install libpcl-dev pcl-tools
```

### Python 依赖（预处理）

```bash
conda create -n libim python=3.10
conda activate libim
pip install -r requirements.txt
```

**requirements.txt 内容**：

```
open3d==0.19.0
easydict==1.10
numpy==1.26.4
matplotlib==3.10.0
scipy==1.15.1
numba==0.61.0
shapely==2.0.6
scikit-image==0.25.1
opencv-python-headless==4.10.0.84
```

### 编译

```bash
# 1. 编译第三方库 point-sam
mkdir -p Thirdparty/point-sam/build
cd Thirdparty/point-sam/build
cmake ..
make -j$(nproc)
cd ../../..

# 2. 编译主项目
mkdir build && cd build
cmake ..
make -j$(nproc)
cd ..
```

编译成功后可执行文件位于 `bin/` 目录下。

---

## 数据准备

本模块使用 **LiBIM-UST** 数据集进行测试。详细的数据准备流程请参阅 [docs/prepare.md](docs/prepare.md)。

**快速开始**：

```bash
# 设置数据集路径
export LiBIM_UST_ROOT="/path/to/LiBIM-UST/"

# Step 1: 生成子图 + 平面分割
python examples/python/data_process/submap3d_generator.py
./Thirdparty/point-sam/build/plane_seg

# Step 2: 生成基准数据（线段 + GT 位姿）
python examples/python/data_process/make_benchmarks.py
```

> 💡 如不想等待预处理，可下载 [processed 数据](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hhuangce_connect_ust_hk/EYiHE-lKVwdLgLRxfDWMpTwBok3Dk_OjmkSkhejte-FcfA?e=AljpIr) 直接放入 `LiBIM-UST/` 目录。

---

## 运行

### Demo（单次配准可视化）

```bash
export LiBIM_UST_ROOT="/path/to/LiBIM-UST/"

# 默认 demo
bin/demo_reg

# 指定子图索引
bin/demo_reg 0

# 指定配置文件和子图索引
bin/demo_reg /configs/interval/15m/2F/2f_office_01.yaml 0
```

### 基准测试（批量评估）

```bash
export LiBIM_UST_ROOT="/path/to/LiBIM-UST/"

# 单序列评估
bin/reg_bm benchmark /configs/interval/15m/2F/building_day.yaml

# 全部序列评估
bash examples/scripts/benchmark.sh
```

---

## 配置文件说明

配置文件位于 `configs/` 目录，以 YAML 格式组织。示例 (`configs/interval/15m/2F/building_day.yaml`)：

| 参数 | 说明 |
|------|------|
| `seq` | 序列名称 |
| `interval` | 子图间隔（米） |
| `max_side_length` | 最大边长 |
| `length_res` / `angle_res` | 描述子量化分辨率 |
| `xy_res` / `yaw_res` | Hough 空间分辨率 |
| `top_l` / `top_k` / `top_j` | Hough 投票 Top-K 参数 |
| `grid_size` / `sigma` / `sdf_dist` | 验证参数 |

---

## 参考文档

- [安装指南](docs/install.md)
- [数据准备](docs/prepare.md)
- [Demo 运行说明](docs/demo.md)
- [基准测试](docs/benchmark.md)

## 致谢

本模块由 **Zhijian QIAO**（香港科技大学 UAV Group）开发。
数据集详见 [SLABIM](https://github.com/HKUST-Aerial-Robotics/SLABIM)。
