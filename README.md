# PointCloud2CAD

> **基于开放词汇语义分割的室内三维点云语义实例分割与 CAD 平面图生成**

本项目实现了一套从 LiDAR 扫描点云到 2D CAD 平面图的自动化处理管线 (Pipeline)，核心思路是利用 2D 基础视觉模型（RAM + GroundingDINO + SAM）对多视角图像进行开放词汇语义分割，再通过自研的多帧 2D→3D 投影融合算法将语义标签迁移至三维点云，最终提取建筑结构生成 CAD 平面图。

---

## 项目结构

```
PointCloud2CAD/
├── rma-dino-sam/               # 阶段一：2D 开放词汇语义分割 (Python)
│   ├── main.py                 # 主入口脚本
│   ├── clip/                   # OpenAI CLIP 模型
│   ├── recognize-anything/     # RAM 图像标签识别
│   ├── GroundingDINO/          # Grounding DINO 开放集目标检测
│   ├── segment_anything/       # SAM 分割一切模型
│   └── EfficientSAM/          # 高效 SAM 变体 (可选)
│
├── PointCloud_Segement_v0/     # 阶段二：3D 语义实例融合 (C++)
│   ├── main.cpp                # 主入口
│   ├── src/                    # 核心源码
│   │   ├── mapping/            # 语义建图：体素哈希、贝叶斯标签融合、实例管理
│   │   ├── cluster/            # 图优化 (PoseGraph)
│   │   ├── sgloop/             # 场景图
│   │   └── tools/              # IO、计时、可视化工具
│   ├── cmake/                  # CMake 依赖配置
│   └── test_data/              # 示例数据 (需自行准备，见下方说明)
│
└── LiDAR2BIM-Registration/     # 阶段三：点云配准与 CAD 生成 (C++/Python)
    ├── preprocess/             # Python 预处理脚本
    │   ├── geometry/           # pc2img 投影、lineseg 线段提取
    │   └── datasets/           # 数据集管理
    ├── src/                    # C++ 配准算法
    ├── include/                # 头文件
    ├── Thirdparty/             # 第三方库 (backward-cpp, nanoflann, point-sam)
    └── configs/                # 场景配置文件
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
│   墙面提取 → 线段拟合 → CAD  │
└──────────────────────────────┘
```

---

## 环境配置

### 阶段一：rma-dino-sam (Python)

**推荐环境**：Python 3.11 + PyTorch ≥ 2.6（CPU 或 GPU）

```bash
# 1. 创建 conda 环境
conda create -n gsam python=3.11
conda activate gsam

# 2. 安装 PyTorch (参考 https://pytorch.org/get-started/previous-versions/)
pip3 install torch torchvision

# 3. 安装子模块
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
cd recognize-anything && pip install -r requirements.txt && pip install -e . && cd ..

# 4. 安装 CLIP（如 pip 安装 RAM 时 git 连接失败）
cd clip && pip install ftfy regex tqdm && pip install . && cd ..

# 5. 其他依赖
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

**模型权重下载**：

| 模型 | 下载地址 |
|------|---------|
| GroundingDINO (SwinT) | [groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) |
| SAM (ViT-H) | [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| RAM (14M) | [ram_swin_large_14m.pth](https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/ram_swin_large_14m.pth) |
| bert-base-uncased | 夸克网盘: https://pan.quark.cn/s/e6cda95e1ca1 |

> ⚠️ 需要在本地配置 `bert-base-uncased` 路径，详见 `rma-dino-sam/README.md`。

### 阶段二：PointCloud_Segement_v0 (C++)

**依赖库**：

| 库 | 参考版本 |
|----|---------|
| Eigen | 3.3.7 |
| Jsoncpp | 1.7.4 |
| OpenCV | 4.2.0 |
| Open3D | 0.19 |
| TBB | 2020.1 |
| CMake | ≥ 4.0 |

```bash
cd PointCloud_Segement_v0
mkdir build && cd build
cmake ..
make
```

### 阶段三：LiDAR2BIM-Registration (C++/Python)

详见 `LiDAR2BIM-Registration/docs/install.md`。

---

## 数据准备

阶段二所需的输入数据目录结构：

```
/path/to/data/
├── color/
│   ├── left/         # 左相机去畸变 RGB 图像 (.jpg)
│   └── right/        # 右相机去畸变 RGB 图像 (.jpg)
├── colorized.pcd     # LiDAR 点云 (PCD 格式)
├── config.yaml       # 相机内参配置 (fx, fy, cx, cy, w, h)
├── transforms.json   # 图像位姿信息 (由 Studio 导出)
└── prediction_no_augment/
    ├── <timestamp>_label.json   # 阶段一输出的语义标签
    └── <timestamp>_mask.png     # 阶段一输出的分割 mask
```

> `test_data/` 中提供了示例数据结构（不含大文件）。大文件（如 `colorized.pcd`）请通过网盘分发。

---

## 运行流程

### Step 1: 2D 语义分割

```bash
cd rma-dino-sam
python main.py "<输入图像文件夹>" "<输出文件夹>"
```

输出每张图像的 `*_label.json` 和 `*_mask.png`。

### Step 2: 3D 语义实例融合

```bash
cd PointCloud_Segement_v0/build
./Test_main "<数据文件夹路径>" "<输出文件夹路径>"
```

**输出文件说明**：

| 文件 | 说明 |
|------|------|
| `segement.pcd` | 分割后的实例点云（颜色编码实例 ID） |
| `removed.pcd` | 移除已识别物体后的点云 |
| `info.txt` | 实例 → 语义标签映射 |
| `points.txt` | 每个实例的点索引（二进制格式） |
| `config.txt` | 运行配置信息 |

### Step 3: CAD 平面图生成

详见 `LiDAR2BIM-Registration/docs/demo.md`。

---

## 主要贡献

本项目中以下模块为自主开发：

- **体素哈希建图 (VoxelHashMap)** — 高效的 3D 空间索引结构
- **2D → 3D 语义投影** — 利用相机内外参将 2D 分割结果反投影到点云
- **贝叶斯标签融合** — 基于多帧观测的概率语义标签决策
- **多帧数据关联与实例合并** — 跨帧实例 ID 一致性维护
- **墙面提取与 CAD 线段输出** — 从语义点云中提取建筑结构

复用的开源模型：RAM、GroundingDINO、SAM、CLIP、pc2img、lineseg。

---

## 许可证

各子模块遵循其各自的开源许可证（详见子目录中的 LICENSE 文件）。
