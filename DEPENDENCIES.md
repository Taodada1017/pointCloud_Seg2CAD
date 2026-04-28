# PointCloud2CAD 依赖清单

> 本文档记录项目所有外部依赖的名称、版本、来源与用途。
>
> 最后更新：2026-04-28

---

## 目录

- [系统工具链](#系统工具链)
- [阶段一：2D 开放词汇语义分割（Python）](#阶段一2d-开放词汇语义分割python)
  - [Python 环境](#python-环境)
  - [深度学习框架](#深度学习框架)
  - [子模块（源码集成）](#子模块源码集成)
  - [预训练模型权重](#预训练模型权重)
  - [Python 包依赖](#python-包依赖)
- [阶段二：3D 语义实例融合（C++）](#阶段二3d-语义实例融合c)
- [阶段三：点云→CAD 线段提取（C++/Python）](#阶段三点云cad-线段提取cpython)
  - [C++ 依赖](#c-依赖)
  - [第三方库（源码集成）](#第三方库源码集成)
  - [Python 依赖](#python-依赖)

---

## 系统工具链

| 工具 | 版本要求 | 来源 | 用途 |
|------|---------|------|------|
| **Ubuntu** | ≥ 20.04 | — | 推荐操作系统 |
| **CMake** | ≥ 4.0 | https://cmake.org/download/ | C++ 构建系统 |
| **GCC** | ≥ 9 (C++17) | `apt: build-essential` | C++ 编译器 |
| **Git** | — | `apt: git` | 版本控制 |
| **Conda** | — | https://docs.conda.io/en/latest/miniconda.html | Python 环境管理 |

---

## 阶段一：2D 开放词汇语义分割（Python）

> 对应目录：`rma-dino-sam/`

### Python 环境

| 项 | 版本 | 来源 |
|----|------|------|
| **Python** | 3.11 | Conda: `conda create -n gsam python=3.11` |

### 深度学习框架

| 包 | 版本要求 | 来源 | 备注 |
|----|---------|------|------|
| **PyTorch (torch)** | ≥ 2.6 | https://pytorch.org/get-started/locally/ | CPU 或 CUDA 版本 |
| **torchvision** | 与 torch 版本匹配 | 同上 | — |

### 子模块（源码集成）

以下子模块以源码形式包含在 `rma-dino-sam/` 目录中，通过 `pip install -e .` 安装：

| 模块 | 版本 | 源码仓库 | 安装方式 |
|------|------|---------|---------|
| **Segment Anything (SAM)** | 1.0 | https://github.com/facebookresearch/segment-anything | `pip install -e segment_anything` |
| **GroundingDINO** | 0.1.0 | https://github.com/IDEA-Research/GroundingDINO | `pip install --no-build-isolation -e GroundingDINO` |
| **RAM (Recognize Anything)** | 0.0.1 | https://github.com/xinyu1205/recognize-anything | `cd recognize-anything && pip install -r requirements.txt && pip install -e .` |
| **OpenAI CLIP** | 1.0 | https://github.com/openai/CLIP | `cd clip && pip install .` |

### 预训练模型权重

| 模型 | 文件名 | 下载地址 |
|------|--------|---------|
| **GroundingDINO (SwinT-OGC)** | `groundingdino_swint_ogc.pth` | https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth |
| **SAM (ViT-H)** | `sam_vit_h_4b8939.pth` | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth |
| **RAM (Swin-Large 14M)** | `ram_swin_large_14m.pth` | https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/ram_swin_large_14m.pth |
| **bert-base-uncased** | 目录 | https://huggingface.co/google-bert/bert-base-uncased （或 [夸克网盘](https://pan.quark.cn/s/e6cda95e1ca1)） |

> 三个模型合集夸克网盘：https://pan.quark.cn/s/2fd76d205f7a

### Python 包依赖

#### GroundingDINO 依赖 (`rma-dino-sam/GroundingDINO/requirements.txt`)

| 包 | 版本 | 来源 (PyPI) |
|----|------|------------|
| torch | — | https://pypi.org/project/torch/ |
| torchvision | — | https://pypi.org/project/torchvision/ |
| transformers | **推荐 4.38.2**（≥4.40 可能报错） | https://pypi.org/project/transformers/ |
| addict | — | https://pypi.org/project/addict/ |
| yapf | — | https://pypi.org/project/yapf/ |
| timm | — | https://pypi.org/project/timm/ |
| numpy | — | https://pypi.org/project/numpy/ |
| opencv-python | — | https://pypi.org/project/opencv-python/ |
| supervision | — | https://pypi.org/project/supervision/ |
| pycocotools | — | https://pypi.org/project/pycocotools/ |

#### RAM 依赖 (`rma-dino-sam/recognize-anything/requirements.txt`)

| 包 | 版本 | 来源 (PyPI) |
|----|------|------------|
| **timm** | **==0.4.12** | https://pypi.org/project/timm/0.4.12/ |
| transformers | ≥4.25.1 | https://pypi.org/project/transformers/ |
| fairscale | ==0.4.4 | https://pypi.org/project/fairscale/0.4.4/ |
| pycocoevalcap | — | https://pypi.org/project/pycocoevalcap/ |
| torch | — | https://pypi.org/project/torch/ |
| torchvision | — | https://pypi.org/project/torchvision/ |
| Pillow | — | https://pypi.org/project/Pillow/ |
| scipy | — | https://pypi.org/project/scipy/ |

> ⚠️ **timm==0.4.12 是 RAM 硬依赖**，不可随意升级。

#### CLIP 依赖 (`rma-dino-sam/clip/requirements.txt`)

| 包 | 版本 | 来源 (PyPI) |
|----|------|------------|
| ftfy | — | https://pypi.org/project/ftfy/ |
| packaging | — | https://pypi.org/project/packaging/ |
| regex | — | https://pypi.org/project/regex/ |
| tqdm | — | https://pypi.org/project/tqdm/ |
| torch | — | https://pypi.org/project/torch/ |
| torchvision | — | https://pypi.org/project/torchvision/ |

#### 其他手动安装的包

| 包 | 来源 (PyPI) | 用途 |
|----|------------|------|
| opencv-python | https://pypi.org/project/opencv-python/ | 图像处理 |
| pycocotools | https://pypi.org/project/pycocotools/ | COCO 格式工具 |
| matplotlib | https://pypi.org/project/matplotlib/ | 可视化 |
| onnxruntime | https://pypi.org/project/onnxruntime/ | ONNX 推理 |
| onnx | https://pypi.org/project/onnx/ | ONNX 模型格式 |

---

## 阶段二：3D 语义实例融合（C++）

> 对应目录：`PointCloud_Segement_v0/`

| 库 | 版本要求 | 来源 | 安装方式 | 用途 |
|----|---------|------|---------|------|
| **Eigen3** | ≥ 3.3.7 | https://eigen.tuxfamily.org/ | `apt: libeigen3-dev` | 线性代数 |
| **OpenCV** | ≥ 4.2.0 | https://opencv.org/ | `apt: libopencv-dev` | 图像处理 |
| **Open3D** | 0.19 | https://www.open3d.org/docs/release/cpp_project.html | 源码编译或预编译包 | 3D 点云处理 |
| **TBB (Threading Building Blocks)** | ≥ 2020.1 | https://github.com/oneapi-src/oneTBB | `apt: libtbb-dev` | 多线程并行 |
| **JsonCpp** | ≥ 1.7.4 | https://github.com/open-source-parsers/jsoncpp | `apt: libjsoncpp-dev` | JSON 解析 |

---

## 阶段三：点云→CAD 线段提取（C++/Python）

> 对应目录：`l2bim/`

### C++ 依赖

| 库 | 版本要求 | 来源 | 安装方式 | 用途 |
|----|---------|------|---------|------|
| **PCL (Point Cloud Library)** | — | https://pointclouds.org/ | `apt: libpcl-dev pcl-tools` | 点云处理 |
| **Eigen3** | ≥ 3.3.7 | https://eigen.tuxfamily.org/ | `apt: libeigen3-dev` | 线性代数（与阶段二共享） |
| **OpenCV** | ≥ 4.2.0 | https://opencv.org/ | `apt: libopencv-dev` | 图像处理（含 ximgproc 模块） |
| **Boost** | — | https://www.boost.org/ | `apt: libboost-dev` | 文件系统、正则、线程等 |
| **OpenMP** | — | 编译器内置 | `apt: libomp-dev` | CPU 并行加速 |
| **yaml-cpp** | — | https://github.com/jbeder/yaml-cpp | `apt: libyaml-cpp-dev` | YAML 配置解析 |
| **glog** | — | https://github.com/google/glog | `apt: libgoogle-glog-dev` | 日志 |
| **Ceres Solver** | — | http://ceres-solver.org/ | `apt: libceres-dev` | 非线性优化 |
| **CGAL** | — | https://www.cgal.org/ | `apt: libcgal-dev` | 计算几何 |
| **GMP** | — | https://gmplib.org/ | `apt: libgmp-dev` | 高精度算术（CGAL 依赖） |
| **MPFR** | — | https://www.mpfr.org/ | `apt: libmpfr-dev` | 多精度浮点（CGAL 依赖） |
| **TBB** | — | https://github.com/oneapi-src/oneTBB | `apt: libtbb-dev` | 多线程（与阶段二共享） |

### 第三方库（源码集成）

以下库以源码形式包含在 `l2bim/Thirdparty/` 目录中，随项目一起编译：

| 库 | 版本 | 源码仓库 | 说明 |
|----|------|---------|------|
| **nanoflann** | 1.5.5 | https://github.com/jlblancoc/nanoflann | Header-only KD-Tree 库 |
| **point-sam** | — | 项目内部 | 点云平面分割模块 |
| **backward-cpp** | — | https://github.com/bombela/backward-cpp | 堆栈跟踪（可选，`USE_BACKWARD=ON`） |

### Python 依赖

> 阶段三 Step 2 墙体正则化脚本 + `l2bim/requirements.txt`

| 包 | 版本 | 来源 (PyPI) | 用途 |
|----|------|------------|------|
| open3d | 0.19.0 | https://pypi.org/project/open3d/0.19.0/ | 3D 处理 |
| opencv-python | — | https://pypi.org/project/opencv-python/ | 图像处理 |
| numpy | 1.26.4 | https://pypi.org/project/numpy/1.26.4/ | 数值计算 |
| scipy | 1.15.1 | https://pypi.org/project/scipy/1.15.1/ | 科学计算 |
| matplotlib | 3.9.2 | https://pypi.org/project/matplotlib/3.9.2/ | 可视化 |
| shapely | 2.0.6 | https://pypi.org/project/shapely/2.0.6/ | 几何运算 |
| scikit-image | 0.25.1 | https://pypi.org/project/scikit-image/0.25.1/ | 图像分析 |
| numba | 0.61.0 | https://pypi.org/project/numba/0.61.0/ | JIT 加速 |
| easydict | 1.10 | https://pypi.org/project/easydict/1.10/ | 字典增强 |
| opencv-python-headless | 4.10.0.84 | https://pypi.org/project/opencv-python-headless/4.10.0.84/ | 无 GUI OpenCV |

---

## 一键安装参考

### Ubuntu apt 依赖（阶段二 + 阶段三 C++）

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git wget curl \
    libeigen3-dev libjsoncpp-dev libtbb-dev libopencv-dev \
    libpcl-dev pcl-tools libboost-dev libomp-dev \
    libyaml-cpp-dev libgoogle-glog-dev libceres-dev \
    libcgal-dev libgmp-dev libmpfr-dev
```

### Python 依赖（阶段一）

```bash
conda create -n gsam python=3.11 -y && conda activate gsam
pip install torch torchvision  # 或 GPU 版本，见 https://pytorch.org
cd rma-dino-sam
pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
cd recognize-anything && pip install -r requirements.txt && pip install -e . && cd ..
cd clip && pip install ftfy regex tqdm && pip install . && cd ..
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

### Python 依赖（阶段三）

```bash
pip install open3d==0.19.0 opencv-python numpy==1.26.4 scipy==1.15.1 \
    matplotlib==3.9.2 shapely==2.0.6 scikit-image==0.25.1 numba==0.61.0 easydict==1.10
```
