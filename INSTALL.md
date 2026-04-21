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
- [阶段三：LiDAR2BIM-Registration（C++ / Python）](#阶段三lidar2bim-registrationc--python)
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
| **阶段三** LiDAR2BIM-Registration | 不需要 | ✅ 完全 CPU | C++ 使用 OpenMP 并行；Python 预处理也无 GPU 依赖 |

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

## 阶段三：LiDAR2BIM-Registration（C++ / Python）

> 点云配准与 CAD 生成。需要 C++ 和 Python 双环境。

### 3.1 安装系统依赖

```bash
sudo apt-get update
sudo apt install -y \
    libboost-dev \
    libyaml-cpp-dev \
    libomp-dev \
    libgmp-dev \
    libmpfr-dev \
    libpcl-dev \
    pcl-tools \
    libgoogle-glog-dev
```

### 3.2 Python 环境（预处理）

```bash
conda create -n libim python=3.10 -y
conda activate libim
cd LiDAR2BIM-Registration
pip install -r requirements.txt
```

### 3.3 编译

```bash
cd LiDAR2BIM-Registration

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

### 3.4 数据准备

```bash
export LiBIM_UST_ROOT="/path/to/LiBIM-UST/"

# 生成子图 + 平面分割
python examples/python/data_process/submap3d_generator.py
./Thirdparty/point-sam/build/plane_seg

# 生成基准数据
python examples/python/data_process/make_benchmarks.py
```

### 3.5 验证安装

```bash
export LiBIM_UST_ROOT="/path/to/LiBIM-UST/"
bin/demo_reg
```

---

## 依赖版本冲突分析与解决

> 本项目三个阶段使用**独立的运行环境**（两个 Conda 环境 + C++ 编译），大部分依赖互不干扰。但以下几处需要注意：

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
│  Conda 环境: libim (Python 3.10) + 系统 C++  ← 阶段三        │
│  open3d==0.19.0, numpy==1.26.4, scipy==1.15.1               │
│  Boost, PCL, yaml-cpp, glog, OpenMP (系统 C++)               │
└──────────────────────────────────────────────────────────────┘
```

> ✅ 阶段一与阶段三使用**不同的 Conda 环境**，Python 依赖完全隔离，不会冲突。

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

### ⚠️ 冲突点 3：opencv-python vs opencv-python-headless

| 来源 | 包名 |
|------|------|
| 阶段一（gsam 环境） | `opencv-python` |
| 阶段三（libim 环境） | `opencv-python-headless==4.10.0.84` |

**分析**：`opencv-python` 和 `opencv-python-headless` 是**互斥包**（同一环境只能装一个）。但由于阶段一和三使用不同 Conda 环境，**不存在实际冲突**。

> ⚠️ **注意**：如果你尝试把所有 Python 依赖装进同一个环境，这两个包会冲突！请务必保持环境隔离。

### ✅ 不冲突的共享依赖

以下依赖在多个子模块中出现，但不会产生版本冲突：

| 依赖 | 阶段一 | 阶段三 | 是否冲突 | 原因 |
|------|--------|--------|---------|------|
| `scipy` | 无版本号 | ==1.15.1 | ❌ | 不同环境 |
| `numpy` | 无版本号 | ==1.26.4 | ❌ | 不同环境 |
| `torch` | ≥2.6 | 不需要 | ❌ | 阶段三不用 PyTorch |
| `Eigen` (C++) | 系统 `find_package` | 硬编码 `/usr/include/eigen3` | ❌ | 分别编译，不互相影响 |
| `OpenCV` (C++) | 系统 `find_package` | 系统 `find_package` | ❌ | 使用相同系统库 |

### 📌 推荐安装策略

1. **严格使用独立 Conda 环境**：`gsam`（阶段一）和 `libim`（阶段三）绝不混用
2. **阶段一按顺序安装**：SAM → GroundingDINO → RAM → CLIP（RAM 最后装，确保 timm 降级）
3. **锁定 transformers 版本**：`pip install transformers==4.38.2`（避免过新版本兼容问题）
4. **C++ 依赖使用系统包管理器**：`apt-get` (Ubuntu) 或 `brew` (macOS)

---

## 模型权重与数据下载

### 一站式下载清单

| 资源 | 用途 | 下载地址 |
|------|------|---------|
| 三个模型 checkpoint | 阶段一 | [夸克网盘](https://pan.quark.cn/s/2fd76d205f7a) |
| bert-base-uncased | 阶段一 | [夸克网盘](https://pan.quark.cn/s/e6cda95e1ca1) |
| LiBIM-UST 数据集 | 阶段三 | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hhuangce_connect_ust_hk/ERoCJi5Q5EZGudr8pQYOXsMBjUsin82b0RYxDVmVQmYCDg?e=ecnrwF) |
| LiBIM-UST processed | 阶段三（可选，跳过预处理） | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hhuangce_connect_ust_hk/EYiHE-lKVwdLgLRxfDWMpTwBok3Dk_OjmkSkhejte-FcfA?e=AljpIr) |

---

## 常见问题

### Q: 阶段一推理很慢？

A: 默认使用 CPU 模式。如果有 GPU（≥16GB 显存），可以在 `main.py` 中将 `device = "cpu"` 改为 `device = "cuda"`。12GB 显存不够用。

### Q: CMake 找不到 Open3D？

A: 确保 Open3D 的 C++ 库已正确安装，并设置 `Open3D_DIR`：

```bash
cmake .. -DOpen3D_DIR=/usr/local/lib/cmake/Open3D
```

### Q: GroundingDINO 编译报错？

A: 确保使用了 `--no-build-isolation` 参数：

```bash
pip install --no-build-isolation -e GroundingDINO
```

### Q: 能否在 Windows 上运行？

A: 阶段一（Python）可以在 Windows 上运行。阶段二和三的 C++ 代码主要在 Linux 下测试，Windows 下需要额外配置（推荐使用 WSL2）。

### Q: 能否在 macOS 上运行？

A: 阶段一（Python / CPU 模式）可以在 macOS 上运行。阶段二需要安装 macOS 版本的依赖库（通过 Homebrew）。阶段三的 PCL 和 glog 在 macOS 上可通过 Homebrew 安装。

---

## 完整依赖版本清单

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

### 阶段三：LiDAR2BIM-Registration — Conda 环境 `libim` (Python 3.10) + 系统 C++

**Python 依赖**（`requirements.txt` 精确锁版本）：

| 包 | 版本 | 备注 |
|----|------|------|
| Python | **3.10** | Conda 创建 |
| open3d | ==0.19.0 | Python 版（与阶段二 C++ 版不冲突） |
| numpy | ==1.26.4 | — |
| scipy | ==1.15.1 | — |
| matplotlib | ==3.10.0 | — |
| numba | ==0.61.0 | JIT 加速 |
| shapely | ==2.0.6 | 几何计算 |
| scikit-image | ==0.25.1 | 图像处理 |
| opencv-python-headless | ==4.10.0.84 | headless 版（无 GUI） |
| easydict | ==1.10 | 配置字典 |

**C++ 系统依赖**：

| 库 | 安装方式 | 备注 |
|----|---------|------|
| Boost | `libboost-dev` | 通用 C++ 库 |
| Eigen3 | 系统自带（与阶段二共享） | — |
| yaml-cpp | `libyaml-cpp-dev` | YAML 配置解析 |
| glog | `libgoogle-glog-dev` | 日志 |
| PCL | `libpcl-dev` | 点云处理 |
| OpenMP | `libomp-dev` | CPU 并行 |
| OpenCV | 系统自带（与阶段二共享） | — |
| GMP/MPFR | `libgmp-dev libmpfr-dev` | CGAL 可选依赖 |

**第三方库**（源码随项目附带，无需额外下载）：

| 库 | 位置 | 说明 |
|----|------|------|
| backward-cpp | `Thirdparty/backward-cpp/` | C++ 调试回溯 |
| nanoflann | `Thirdparty/nanoflann/` | KD-Tree 快速近邻搜索 |
| point-sam | `Thirdparty/point-sam/` | 平面分割（需编译） |
