# PointCloud2CAD 环境部署指南

> 推荐操作系统：**Ubuntu 20.04+**

---

## 硬件要求

| 项目 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | x86_64，4 核+ | — |
| 内存 | ≥ 16 GB | ≥ 32 GB |
| GPU | 可选（纯 CPU 可运行） | NVIDIA ≥ 16GB 显存 |
| CUDA | — | ≥ 11.8 |
| 磁盘 | ≥ 15 GB | — |

> ⚠️ 12GB 显存不够同时加载 RAM + GroundingDINO + SAM 三个模型。

---

## 前置工具

```bash
# Ubuntu
sudo apt-get update
sudo apt-get install -y build-essential cmake git wget curl
```

| 工具 | 版本要求 |
|------|---------|
| CMake | ≥ 4.0 |
| GCC | ≥ 9 (C++17) |
| Conda | — |

---

## 阶段一：stage1_2d_semantic_seg（Python）

### 1. 创建环境

```bash
conda create -n gsam python=3.11 -y
conda activate gsam
```

### 2. 安装 PyTorch

```bash
# CPU 版本
pip3 install torch torchvision

# GPU 版本（参考 https://pytorch.org/get-started/locally/）
```

> ⚠️ PyTorch 版本必须 **≥ 2.6**

### 3. 安装子模块

```bash
cd stage1_2d_semantic_seg

# SAM
python -m pip install -e segment_anything

# GroundingDINO
pip install --no-build-isolation -e GroundingDINO

# RAM
cd recognize-anything
pip install -r requirements.txt
pip install -e .
cd ..

# CLIP（如 RAM 安装时下载失败）
cd clip && pip install ftfy regex tqdm && pip install . && cd ..

# 其他依赖
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

### 4. 下载模型权重

| 模型 | 文件名 | 下载地址 |
|------|--------|---------|
| GroundingDINO | `groundingdino_swint_ogc.pth` | [GitHub](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) |
| SAM (ViT-H) | `sam_vit_h_4b8939.pth` | [Meta AI](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| RAM (14M) | `ram_swin_large_14m.pth` | [HuggingFace](https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/ram_swin_large_14m.pth) |
| bert-base-uncased | 目录 | [夸克网盘](https://pan.quark.cn/s/e6cda95e1ca1) |

> **三个模型合集**：[夸克网盘](https://pan.quark.cn/s/2fd76d205f7a)

### 5. 配置路径

**权重路径**（环境变量）：

```bash
export WEIGHTS_DIR="/你的路径/weights"
```

**bert-base-uncased 路径**：

- 修改 `GroundingDINO/groundingdino/util/get_tokenlizer.py`：
  ```python
  tokenizer = AutoTokenizer.from_pretrained("/你的路径/weights/bert-base-uncased")
  ```

- 修改 `recognize-anything/ram/models/utils.py` 第 130 行：
  ```python
  tokenizer = BertTokenizer.from_pretrained("/你的路径/weights/bert-base-uncased")
  ```

### 6. CPU/GPU 切换

修改 `main.py` 第 216 行：

```python
device = "cpu"   # CPU 模式
device = "cuda"  # GPU 模式
```

---

## 阶段二：stage2_3d_instance_fusion（C++）

### 1. 安装依赖

```bash
# Ubuntu
sudo apt-get install -y libeigen3-dev libjsoncpp-dev libtbb-dev libopencv-dev

# macOS
brew install eigen jsoncpp tbb opencv open3d cmake
```

**Open3D 0.19**（C++ 库）：参考 https://www.open3d.org/docs/release/cpp_project.html

### 2. 编译

```bash
cd stage2_3d_instance_fusion
mkdir build && cd build
cmake ..
make -j$(nproc)
```

> 如找不到 Open3D：`cmake .. -DOpen3D_DIR=/path/to/open3d/lib/cmake/Open3D`

---

## 阶段三：stage3_cad_generation（C++ / Python）

### Step 1：C++ 投影

**依赖**：

```bash
# Ubuntu
sudo apt-get install -y libpcl-dev pcl-tools libeigen3-dev libopencv-dev libboost-dev libomp-dev

# macOS
brew install pcl eigen opencv boost libomp cmake
```

**编译**：

```bash
cd stage3_cad_generation
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make pcd_projection
```

### Step 2：Python 规则化

```bash
pip install opencv-python numpy
```

> Python ≥ 3.8 即可，无需 GPU。

---

## 依赖版本清单

### 阶段一（Python）

| 包 | 版本要求 | 备注 |
|----|---------|------|
| Python | 3.11 | — |
| PyTorch | **≥ 2.6** | 必须 |
| timm | **==0.4.12** | RAM 硬依赖 |
| transformers | **推荐 4.38.2** | ≥4.40 可能报错 |

### 阶段二（C++）

| 库 | 版本 |
|----|------|
| Eigen3 | 3.3.7+ |
| OpenCV | 4.2.0+ |
| Open3D | 0.19 |
| TBB | 2020.1+ |
| Jsoncpp | 1.7.4+ |

### 阶段三（C++）

| 库 | 说明 |
|----|------|
| PCL | Point Cloud Library |
| Eigen3 | 与阶段二共享 |
| OpenCV | 与阶段二共享 |
| Boost | — |
| OpenMP | CPU 并行 |

---

## 常见问题

| 问题 | 解决方案 |
|------|---------|
| `torch.load` 版本错误 | 升级 PyTorch ≥ 2.6 |
| `apply_chunking_to_forward` 导入错误 | `pip install transformers==4.38.2` |
| CMake 找不到 Open3D | `cmake .. -DOpen3D_DIR=/usr/local/lib/cmake/Open3D` |
| CMake 找不到 PCL | `sudo apt-get install libpcl-dev pcl-tools` |
| GroundingDINO 编译报错 | 使用 `pip install --no-build-isolation -e GroundingDINO` |
