# PointCloud2CAD 环境部署指南

> 本文档提供完整的跨设备安装部署流程，覆盖 Pipeline 的三个阶段。
> 推荐操作系统：**Ubuntu 20.04+**（阶段二和三依赖 Linux C++ 工具链）。

---

## 目录

- [设备与硬件要求](#设备与硬件要求)
- [CPU / GPU 运行说明](#cpu--gpu-运行说明)
- [前置要求](#前置要求)
- [阶段一：rma-dino-sam（Python）](#阶段一rma-dino-sampython)
- [阶段二：PointCloud_Segement_v0（C++）](#阶段二pointcloud_segement_v0c)
- [阶段三：l2bim（C++ / Python）](#阶段三l2bimc--python)
- [依赖版本冲突分析与解决](#依赖版本冲突分析与解决)
- [模型权重与数据下载](#模型权重与数据下载)
- [常见问题](#常见问题)
- [完整依赖版本清单](#完整依赖版本清单)

---

## 设备与硬件要求

### 最低配置

| 项目 | 要求 | 说明 |
|------|------|------|
| **操作系统** | Ubuntu 20.04+ (推荐) | macOS 可运行阶段一二；Windows 推荐 WSL2 |
| **CPU** | x86_64，4 核+ | ARM (Apple Silicon) 需从源码编译 C++ 依赖 |
| **内存** | ≥ 16 GB | 阶段一加载 3 个模型约需 8-10GB；阶段二处理大点云时内存消耗较高 |
| **磁盘空间** | ≥ 15 GB | 模型权重 ~5GB + 编译产物 + 点云数据 |
| **GPU** | **可选** | 纯 CPU 即可运行全部三个阶段（见下方说明） |

### 推荐配置（高效运行）

| 项目 | 要求 |
|------|------|
| **GPU** | NVIDIA GPU，≥ 16GB 显存（如 RTX 3090/4090/A100） |
| **CUDA** | ≥ 11.8（配合 PyTorch ≥ 2.6） |
| **内存** | ≥ 32 GB |

> ⚠️ 12GB 显存（如 RTX 3060）**不够**同时加载 RAM + GroundingDINO + SAM 三个模型。

---

## CPU / GPU 运行说明

### 整个 Pipeline 纯 CPU 就能跑吗？✅ 可以！

| 阶段 | GPU 需求 | 纯 CPU 支持 | 说明 |
|------|---------|-------------|------|
| **阶段一** rma-dino-sam | 可选 | ✅ 支持 | `main.py` 默认 `device = "cpu"`；GroundingDINO 在无 CUDA 时自动编译 CPU 版本 |
| **阶段二** PointCloud_Segement_v0 | 不需要 | ✅ 完全 CPU | 纯 C++ 项目，使用 TBB 做 CPU 多线程并行 |
| **阶段三** l2bim | 不需要 | ✅ 完全 CPU | Step 1 (C++) 使用 OpenMP 并行；Step 2 (Python) 无 GPU 依赖 |

### CPU vs GPU 性能对比（阶段一）

| 运行环境 | 单张图推理时间（估算） | 100 张图总时间 |
|---------|----------------------|---------------|
| NVIDIA GPU ≥16GB 显存 | ~3-5 秒/张 | ~5-8 分钟 |
| CPU (现代 i7/Ryzen 7) | ~30-120 秒/张 | ~50 分钟 - 3 小时 |
| Apple M1/M2 (CPU 模式) | ~40-90 秒/张 | ~1-2.5 小时 |

### 如何切换 CPU / GPU

修改 `rma-dino-sam/main.py` 第 216 行：

```python
# CPU 模式（默认）
device = "cpu"

# GPU 模式（需要 NVIDIA GPU + CUDA）
device = "cuda"
```

> 💡 阶段二和阶段三**没有 GPU 选项**，它们天然就是 CPU 程序。性能取决于 CPU 核心数和内存大小。

---

## 前置要求

| 工具 | 最低版本 | 说明 |
|------|---------|------|
| **Git** | — | 代码管理 |
| **Conda** (Miniconda/Anaconda) | — | Python 环境管理 |
| **CMake** | ≥ 4.0 | C++ 构建系统 |
| **Make** / **GCC** | Make ≥ 4.2, GCC ≥ 9 | C++17 编译器 |
| **磁盘空间** | ≥ 15 GB | 模型权重 ~5GB + 数据 ~400MB+ |

```bash
# Ubuntu 安装基础工具
sudo apt-get update
sudo apt-get install -y build-essential cmake git wget curl
```

---

## 阶段一：rma-dino-sam（Python）

> 2D 开放词汇语义分割。推荐环境：Python 3.11 + PyTorch ≥ 2.6。

### 1.1 创建 Conda 环境

```bash
conda create -n gsam python=3.11 -y
conda activate gsam
```

> 💡 如果 conda 下载慢，可使用国内镜像源：
> ```bash
> conda create -n gsam python=3.11 -c https://mirrors.ustc.edu.cn/anaconda/pkgs/main/ -y
> ```

### 1.2 安装 PyTorch

```bash
# CPU 版本
pip3 install torch torchvision

# GPU 版本（需要 CUDA，推荐 ≥16GB 显存）
# 参考 https://pytorch.org/get-started/locally/ 选择对应版本
```

> ⚠️ **PyTorch 版本必须 ≥ 2.6**，低版本会报 `ValueError: Due to a serious vulnerability issue in torch.load`。

### 1.3 安装子模块

**在 `rma-dino-sam/` 目录下执行**：

```bash
cd rma-dino-sam

# 安装 SAM
python -m pip install -e segment_anything

# 安装 GroundingDINO
pip install --no-build-isolation -e GroundingDINO

# 安装 RAM
cd recognize-anything
pip install -r requirements.txt
pip install -e .
cd ..

# 安装 CLIP（如上一步 git 下载失败）
cd clip
pip install ftfy regex tqdm
pip install .
cd ..

# 安装其他依赖
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

> ⚠️ RAM 的 `requirements.txt` 最后一行会尝试从 GitHub 下载 CLIP。如果网络连接失败，需要手动安装本地的 `clip/` 目录（如上述步骤 4）。安装好后可删除 `requirements.txt` 中 CLIP 的那行再重新执行。

### 1.4 下载模型权重

| 模型 | 文件名 | 大小 | 下载地址 |
|------|--------|------|---------|
| GroundingDINO | `groundingdino_swint_ogc.pth` | ~694MB | [GitHub](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) |
| SAM (ViT-H) | `sam_vit_h_4b8939.pth` | ~2.4GB | [Meta AI](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| RAM (14M) | `ram_swin_large_14m.pth` | ~1.5GB | [HuggingFace](https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/ram_swin_large_14m.pth) |
| bert-base-uncased | `bert-base-uncased/` 目录 | ~420MB | [夸克网盘](https://pan.quark.cn/s/e6cda95e1ca1) |

> **三个模型的 checkpoint 合集**：[夸克网盘](https://pan.quark.cn/s/2fd76d205f7a)

建议将所有权重文件统一放在一个目录下（如 `~/weights/`）：

```
~/weights/
├── groundingdino_swint_ogc.pth
├── sam_vit_h_4b8939.pth
├── ram_swin_large_14m.pth
└── bert-base-uncased/
    ├── config.json
    ├── vocab.txt
    └── ...
```

### 1.5 配置本地路径

**必须完成以下配置**，否则代码会尝试联网下载并报 443 错误。

#### (a) 配置模型权重路径

`main.py` 现在通过 **环境变量 `WEIGHTS_DIR`** 配置权重目录，无需再手动改代码：

```bash
# 在运行前设置环境变量指向你的权重目录
export WEIGHTS_DIR="/你的路径/weights"
```

如果不设置环境变量，默认在 `rma-dino-sam/weights/` 目录下查找。

> 💡 三个权重文件需放在同一个目录下：
> - `ram_swin_large_14m.pth`
> - `groundingdino_swint_ogc.pth`
> - `sam_vit_h_4b8939.pth`

#### (b) 配置 bert-base-uncased 路径

**DINO** — 修改 `GroundingDINO/groundingdino/util/get_tokenlizer.py`：

```python
# 在文件顶部添加：
local_path = "/你的路径/weights/bert-base-uncased"

# 修改 get_tokenlizer 函数中：
tokenizer = AutoTokenizer.from_pretrained(local_path)
```

**RAM** — 修改 `recognize-anything/ram/models/utils.py` 第 130 行的 `init_tokenizer` 函数：

```python
tokenizer = BertTokenizer.from_pretrained("/你的路径/weights/bert-base-uncased")
```

### 1.6 已知问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `ImportError: cannot import name 'apply_chunking_to_forward'` | transformers 版本兼容性 | [参考修复](https://blog.csdn.net/m0_75101512/article/details/153682726)，修改 `recognize-anything/ram/models/bert.py` 第 28 行 |
| `torch.load` 版本错误 | PyTorch < 2.6 | 升级 PyTorch 到 ≥ 2.6 |
| RAM++ 推理失败 | 仅支持 RAM (14M) | 使用 `ram_swin_large_14m.pth` |

### 1.7 验证安装

```bash
cd rma-dino-sam
python main.py "test_images/" "test_output/"
```

如果能成功处理图像并输出 `_mask.png` 和 `_label.json`，则安装成功。

---

## 阶段二：PointCloud_Segement_v0（C++）

> 3D 语义实例融合。纯 C++ 环境。

### 2.1 安装依赖库

#### Ubuntu

```bash
sudo apt-get update
sudo apt-get install -y \
    libeigen3-dev \
    libjsoncpp-dev \
    libtbb-dev \
    cmake
```

**OpenCV 4.2+**：

```bash
sudo apt-get install -y libopencv-dev
# 或从源码编译（如果需要特定版本）
```

**Open3D 0.19**：

```bash
# 推荐从源码编译或使用预编译包
# 参考 https://www.open3d.org/docs/release/compilation.html
# 或使用 pip 安装（仅 Python 接口，C++ 需要源码编译）
```

> 💡 Open3D 的 C++ 安装参考：https://www.open3d.org/docs/release/cpp_project.html

#### macOS (Homebrew)

```bash
brew install eigen jsoncpp tbb opencv open3d cmake
```

### 2.2 编译

```bash
cd PointCloud_Segement_v0
mkdir build && cd build
cmake ..
make -j$(nproc)
```

> ⚠️ 如果 CMake 找不到某些库，需手动设置路径。例如：
> ```bash
> cmake .. -DOpen3D_DIR=/path/to/open3d/lib/cmake/Open3D
> ```

### 2.3 验证安装

```bash
./Test_main "path/to/test_data" "path/to/output"
```

---

## 阶段三：l2bim（C++ / Python）

> 点云 → 2D CAD 直线段提取。分为 Step 1 (C++ 投影) 和 Step 2 (Python 规则化) 两步。

### 3.1 Step 1 依赖：C++ 系统库

```bash
# Ubuntu
sudo apt-get update
sudo apt-get install -y \
    libpcl-dev \
    pcl-tools \
    libeigen3-dev \
    libopencv-dev \
    libboost-dev \
    libomp-dev
```

```bash
# macOS (Homebrew)
brew install pcl eigen opencv boost libomp cmake
```

> **注意**：阶段二使用 **Open3D** 处理点云，阶段三 Step 1 使用 **PCL** (Point Cloud Library)，两者是不同的点云库。

### 3.2 Step 2 依赖：Python

Step 2 (`wall_regularization_v2.py`) 是轻量 Python 脚本，仅需：

```bash
pip install opencv-python numpy
```

> 可在任意 Python 3.8+ 环境中运行，无需 GPU，无需 Conda 专属环境。

### 3.3 编译

```bash
cd l2bim

# 1. （可选）编译第三方库 point-sam
mkdir -p Thirdparty/point-sam/build
cd Thirdparty/point-sam/build
cmake ..
make -j$(nproc)
cd ../../..

# 2. 编译主项目（只需构建 pcd_projection 目标）
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make pcd_projection
cd ..
```

Windows：

```powershell
cd l2bim
cmake -S . -B build_vs17_x64 -G "Visual Studio 17 2022" -A x64
cmake --build build_vs17_x64 --config Release --target pcd_projection
```

### 3.4 验证安装

```bash
# Step 1: C++ 投影
cd l2bim
./bin/pcd_projection pcds/001.pcd output/001

# Step 2: Python 规则化
python examples/python/wall_regularization_v2.py \
    --input output/001/projection_lines_refined.png \
    --output-dir output/001
```

如果 `output/001/` 下生成了 `projection.png` 和 `v2_step8_final_result.png`，则安装成功。

---

## 依赖版本冲突分析与解决

> 本项目三个阶段使用**独立的运行环境**（Conda 环境 + C++ 编译），大部分依赖互不干扰。但以下几处需要注意：

### 环境隔离概览

```
┌──────────────────────────────────────────────────────────────┐
│  Conda 环境: gsam (Python 3.11)  ← 阶段一                    │
│  PyTorch ≥2.6, timm==0.4.12, transformers, SAM, RAM, DINO   │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  系统 C++ 工具链  ← 阶段二                                    │
│  Eigen, OpenCV, Open3D (C++), Jsoncpp, TBB                  │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  系统 C++ 工具链 + 任意 Python 3.8+  ← 阶段三                │
│  C++: PCL, Eigen, OpenCV, Boost, OpenMP                      │
│  Python: opencv-python, numpy                                │
└──────────────────────────────────────────────────────────────┘
```

> ✅ 阶段一使用独立 Conda 环境 `gsam`，与阶段三 Python 依赖无冲突。
> ✅ 阶段三 Python 仅需 `opencv-python` + `numpy`，可在阶段一的 `gsam` 环境中直接运行，也可在独立环境中运行。

### ⚠️ 冲突点 1：timm 版本（阶段一内部）

| 来源 | 要求 |
|------|------|
| `recognize-anything/requirements.txt` | `timm==0.4.12`（精确锁定旧版） |
| `GroundingDINO/requirements.txt` | `timm`（无版本号，默认装最新） |

**问题**：RAM 模型对 `timm==0.4.12` 有硬依赖（使用了旧版 `timm.models.vision_transformer` API）。如果先装了新版 timm，RAM 推理会报 `AttributeError`。

**解决方案**：按照文档推荐的安装顺序（先 GroundingDINO → 再 RAM），RAM 的 `pip install -r requirements.txt` 会自动将 timm **降级到 0.4.12**。GroundingDINO 对 timm 版本不敏感，降级后不影响使用。

```bash
# ✅ 正确顺序（已在文档 1.3 中体现）
pip install --no-build-isolation -e GroundingDINO   # 装了最新 timm
cd recognize-anything && pip install -r requirements.txt  # timm 降级到 0.4.12
```

### ⚠️ 冲突点 2：transformers 版本上限（阶段一内部）

| 来源 | 要求 |
|------|------|
| `recognize-anything/requirements.txt` | `transformers>=4.25.1` |
| `GroundingDINO/requirements.txt` | `transformers`（无版本约束） |

**问题**：两者都没有设版本上限，pip 会安装最新版。但 transformers **≥4.40** 可能导致 RAM 内部的 `bert.py` 报错：

```
ImportError: cannot import name 'apply_chunking_to_forward' from 'transformers'
```

**解决方案**：安装一个兼容版本，或手动修复 `bert.py`：

```bash
# 方案一：锁定兼容版本
pip install transformers==4.38.2

# 方案二：修复源码（参考 1.6 已知问题）
# 修改 recognize-anything/ram/models/bert.py 第 28 行
```

### ✅ 不冲突的共享依赖

以下依赖在多个阶段中出现，但不会产生版本冲突：

| 依赖 | 阶段一 | 阶段三 | 是否冲突 | 原因 |
|------|--------|--------|---------|------|
| `numpy` | 无版本号 | 无版本号 | ❌ | 兼容 |
| `opencv-python` | 最新 | 最新 | ❌ | 相同包 |
| `Eigen` (C++) | 系统 `find_package` | 系统 `find_package` | ❌ | 共享系统库 |
| `OpenCV` (C++) | 系统 `find_package` | 系统 `find_package` | ❌ | 共享系统库 |

### 📌 推荐安装策略

1. **阶段一使用独立 Conda 环境 `gsam`**
2. **阶段一按顺序安装**：SAM → GroundingDINO → RAM → CLIP（RAM 最后装，确保 timm 降级）
3. **锁定 transformers 版本**：`pip install transformers==4.38.2`（避免过新版本兼容问题）
4. **C++ 依赖使用系统包管理器**：`apt-get` (Ubuntu) 或 `brew` (macOS)
5. **阶段三 Python** 可复用 `gsam` 环境或直接在系统 Python 中运行

---

## 模型权重与数据下载

### 一站式下载清单

| 资源 | 用途 | 下载地址 |
|------|------|---------|
| 三个模型 checkpoint | 阶段一 | [夸克网盘](https://pan.quark.cn/s/2fd76d205f7a) |
| bert-base-uncased | 阶段一 | [夸克网盘](https://pan.quark.cn/s/e6cda95e1ca1) |

---

## 常见问题

### Q: 阶段一推理很慢？

A: 默认使用 CPU 模式。如果有 GPU（≥16GB 显存），可以在 `main.py` 中将 `device = "cpu"` 改为 `device = "cuda"`。12GB 显存不够用。

### Q: CMake 找不到 Open3D？

A: 确保 Open3D 的 C++ 库已正确安装，并设置 `Open3D_DIR`：

```bash
cmake .. -DOpen3D_DIR=/usr/local/lib/cmake/Open3D
```

### Q: CMake 找不到 PCL？

A: 确保 PCL 已通过系统包管理器安装：

```bash
# Ubuntu
sudo apt-get install -y libpcl-dev pcl-tools

# macOS
brew install pcl
```

如果仍找不到，手动指定：

```bash
cmake .. -DPCL_DIR=/usr/lib/x86_64-linux-gnu/cmake/pcl
```

### Q: GroundingDINO 编译报错？

A: 确保使用了 `--no-build-isolation` 参数：

```bash
pip install --no-build-isolation -e GroundingDINO
```

### Q: 能否在 Windows 上运行？

A: 阶段一（Python）可以在 Windows 上运行。阶段二和三的 C++ 代码主要在 Linux 下测试，Windows 下需要额外配置（推荐使用 WSL2）。

### Q: 能否在 macOS 上运行？

A: 阶段一（Python / CPU 模式）可以在 macOS 上运行。阶段二需要安装 macOS 版本的依赖库（通过 Homebrew）。阶段三的 PCL 在 macOS 上可通过 Homebrew 安装。

---

## 完整依赖版本清单

### 环境总览

| 阶段 | 环境 | 关键依赖 | 备注 |
|------|------|----------|------|
| 阶段一 | Conda `gsam` (Python 3.11) + PyTorch + GPU | RAM, GroundingDINO, SAM 权重文件 | 可在远程 GPU 服务器运行 |
| 阶段二 | C++ 17 (CMake + Open3D + Eigen + OpenCV) | Open3D (C++) | 需要编译 |
| 阶段三 Step 1 | C++ 17 (CMake + **PCL** + Eigen + OpenCV) | PCL (Point Cloud Library) | 需要编译 |
| 阶段三 Step 2 | Python 3.8+ | `opencv-python`, `numpy` | 轻量依赖 |

> **注意**：阶段二使用 **Open3D** 处理点云，阶段三 Step 1 使用 **PCL**，两者是不同的点云库。

### 阶段一：rma-dino-sam — Conda 环境 `gsam` (Python 3.11)

| 包 | 版本要求 | 来源 | 备注 |
|----|---------|------|------|
| Python | 3.11 | — | Conda 创建 |
| PyTorch | **≥ 2.6** | RAM/DINO/SAM 均需 | <2.6 会报 `torch.load` 安全错误 |
| torchvision | 与 PyTorch 配套 | RAM/DINO/CLIP | — |
| **timm** | **==0.4.12** | RAM 硬依赖 | ⚠️ 不能用新版，见冲突分析 |
| **transformers** | **≥4.25.1, 推荐 4.38.2** | RAM/DINO | ⚠️ ≥4.40 可能报错，见冲突分析 |
| fairscale | ==0.4.4 | RAM | — |
| Pillow | 最新 | RAM | — |
| scipy | 最新 | RAM | — |
| addict | 最新 | GroundingDINO | — |
| yapf | 最新 | GroundingDINO | — |
| supervision | 最新 | GroundingDINO | — |
| opencv-python | 最新 | GroundingDINO / main.py | — |
| pycocotools | 最新 | GroundingDINO / main.py | — |
| matplotlib | 最新 | main.py | — |
| onnxruntime | 最新 | main.py (可选) | — |
| onnx | 最新 | main.py (可选) | — |
| ftfy | 最新 | CLIP | — |
| regex | 最新 | CLIP | — |
| tqdm | 最新 | CLIP | — |
| packaging | 最新 | CLIP | — |

### 阶段二：PointCloud_Segement_v0 — 系统 C++ 工具链

| 库 | 参考版本 | 安装方式 | 备注 |
|----|---------|---------|------|
| CMake | **≥ 4.0** | apt / brew / 源码 | CMakeLists.txt 要求 ≥3.15，推荐 4.0 |
| GCC | **≥ 9** | apt | 需要 C++17 支持 |
| Eigen3 | 3.3.7+ | `libeigen3-dev` | 线性代数 |
| Jsoncpp | 1.7.4+ | `libjsoncpp-dev` | JSON 解析 |
| OpenCV | **4.2.0+** | `libopencv-dev` | 图像处理 |
| Open3D | **0.19** | 源码编译 / brew | ⚠️ 需要 C++ 库，pip 版不够 |
| TBB | 2020.1+ | `libtbb-dev` | CPU 并行 |

### 阶段三：l2bim — 系统 C++ + Python

**Step 1 (C++) 系统依赖**：

| 库 | 安装方式 | 备注 |
|----|---------|------|
| PCL | `libpcl-dev` + `pcl-tools` | 点云处理 (Point Cloud Library) |
| Eigen3 | 系统自带（与阶段二共享） | 线性代数 |
| OpenCV | 系统自带（与阶段二共享） | 图像处理 |
| Boost | `libboost-dev` | 通用 C++ 库 |
| OpenMP | `libomp-dev` | CPU 并行 |

**Step 2 (Python) 依赖**：

| 包 | 版本 | 备注 |
|----|------|------|
| Python | **≥ 3.8** | 任意环境即可 |
| opencv-python | 最新 | 图像处理 |
| numpy | 最新 | 数值计算 |

**第三方库**（源码随项目附带，无需额外下载）：

| 库 | 位置 | 说明 |
|----|------|------|
| nanoflann | `Thirdparty/nanoflann/` | KD-Tree 快速近邻搜索 |
| point-sam | `Thirdparty/point-sam/` | 平面分割（可选，需编译） |

---

> **文档版本**: 2026-04-21
> **适用代码**: PointCloud2CAD 主分支
