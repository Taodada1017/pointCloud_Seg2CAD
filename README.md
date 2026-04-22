# PointCloud2CAD

> **面向室内BIM重建的点云语义分割与CAD平面图自动生成**
>
> Indoor Point Cloud Semantic Segmentation and Automatic CAD Floor Plan Generation for BIM Reconstruction

本项目实现了一套从 LiDAR 扫描点云到 2D CAD 平面图的自动化处理管线，**核心创新**是提出了基于**3D点集级IoU**的多视角语义实例数据关联方法，解决了传统2D IoU在大视角变化下关联失败的问题。

---

## ✨ 核心创新

### 3D点集级IoU数据关联

**问题**：传统方法在2D图像平面计算IoU进行实例关联，当相机视角变化较大时，同一物体的2D mask可能完全不重叠，导致关联失败。

**本文方法**：将数据关联从2D图像空间提升到3D点云空间，直接在点索引集合上计算IoU：

$$\text{IoU}_{3D} = \frac{|points_A \cap points_B|}{|points_A \cup points_B|}$$

**优势**：
- **视角不变性**：同一物体占据的3D空间是固定的，无论从哪个视角观测，3D点集都会有重叠
- **遮挡鲁棒性**：即使部分遮挡，未被遮挡的共同可见点仍可保持关联
- **尺度无关性**：不受相机远近影响，点索引不随距离变化

| 特性 | 传统2D IoU | 本文3D IoU |
|------|-----------|-----------|
| 视角敏感性 | 高度敏感 | 天然鲁棒 |
| 遮挡影响 | 严重 | 有限 |
| 适用场景 | 小视角变化 | 任意视角变化 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              系统输入                                        │
│       彩色点云 (.pcd)  +  多视角RGB图像  +  相机位姿  +  相机内参             │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    阶段一：2D开放词汇语义分割                                  ┃
┃                                                                             ┃
┃   图像 → [关键帧选取] → RAM标签 → [语义先验过滤] → DINO检测 → SAM分割         ┃
┃   输出：mask图像 + label.json                                                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                              ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃               阶段二：3D语义实例融合 ⭐ 核心创新                               ┃
┃                                                                             ┃
┃   ① 体素哈希加速的视锥体点云筛选                                              ┃
┃   ② 2D mask → 3D点集映射（针孔相机反投影）                                    ┃
┃   ③ 3D点集级IoU数据关联 ← 核心创新                                           ┃
┃   ④ 贝叶斯多帧语义融合                                                       ┃
┃   输出：segement.pcd（实例分割） + removed.pcd（纯结构点云）                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                              ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃               阶段三：自适应点云投影与CAD线段提取                               ┃
┃                                                                             ┃
┃   ① Z直方图自动检测地板/天花板高度                                            ┃
┃   ② 多层水平切片投票 → 鲁棒的墙体栅格投影                                     ┃
┃   ③ 骨架化 + HoughLinesP 线段检测 + 共线碎片合并                              ┃
┃   输出：floorplan.dxf + floorplan.svg（带真实尺度的CAD平面图）                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

## 📁 项目结构

```
PointCloud2CAD/
├── rma-dino-sam/                    # 阶段一：2D开放词汇语义分割 (Python)
│   ├── main.py                      # 主入口脚本
│   ├── recognize-anything/          # RAM 图像标签识别
│   ├── GroundingDINO/               # Grounding DINO 开放集目标检测
│   ├── segment_anything/            # SAM 分割一切模型
│   └── clip/                        # OpenAI CLIP 模型
│
├── PointCloud_Segement_v0/          # 阶段二：3D语义实例融合 (C++) ⭐核心
│   ├── main.cpp                     # 主入口
│   ├── src/
│   │   ├── mapping/                 # 体素哈希、3D IoU关联、贝叶斯融合、实例管理
│   │   └── tools/                   # IO、计时、可视化工具
│   ├── cmake/                       # CMake 依赖配置
│   └── test_data/                   # 示例数据
│
├── l2bim/                           # 阶段三：点云→CAD线段提取 (C++/Python)
│   ├── examples/
│   │   ├── cpp/pcd_projection.cpp   # 点云投影 (C++)
│   │   └── python/wall_regularization_v2.py  # 直线段检测 (Python)
│   ├── src/                         # C++ 源码
│   └── Thirdparty/                  # 第三方库
│
├── docs/                            # 技术文档
├── knowledge/                       # 知识文档
├── environment.md                   # 环境部署指南
└── README.md                        # 本文档
```

---

## 🔧 技术特点

- **零样本识别**：基于开放词汇模型（RAM + GroundingDINO + SAM），无需针对特定场景训练
- **视角鲁棒**：3D点集级IoU关联，不受相机视角变化影响
- **自适应参数**：Z直方图自动检测建筑高度，泛化不同场景
- **真实尺度**：输出CAD图带米制坐标，可直接用于BIM建模

---

## 📊 数据格式

### 输入数据目录结构

```
data_folder/
├── color/
│   ├── left/                  # 左相机 RGB 图像 (.jpg)
│   └── right/                 # 右相机 RGB 图像 (.jpg)
├── colorized.pcd              # LiDAR 彩色点云
├── config.yaml                # 相机内参配置
├── transforms.json            # 相机位姿 (4×4变换矩阵)
└── prediction_no_augment/     # 阶段一输出目录
    ├── *_mask.png             # 语义mask (uint16)
    └── *_label.json           # 标签文件
```

### 相机内参配置 (config.yaml)

```yaml
image_width: 1600
image_height: 1600
camera_fx: 785.17
camera_fy: 785.30
camera_cx: 800.00
camera_cy: 800.00
```

### 输出文件

| 阶段 | 输出文件 | 说明 |
|------|---------|------|
| 阶段一 | `*_mask.png`, `*_label.json` | 2D语义mask和标签 |
| 阶段二 | `segement.pcd` | 实例分割点云 |
| 阶段二 | `removed.pcd` | 移除家具后的结构点云 |
| 阶段三 | `floorplan.dxf` | CAD平面图 (AutoCAD格式) |
| 阶段三 | `floorplan.svg` | 矢量平面图 (带比例尺) |

---

## 🚀 快速运行

> 环境搭建详见 **[environment.md](environment.md)**

```bash
# ━━━━ 阶段一：2D语义分割 (GPU推荐) ━━━━
cd rma-dino-sam
export WEIGHTS_DIR="/path/to/weights"
python main.py "<数据目录>/color/left"  "<数据目录>/prediction_no_augment"
python main.py "<数据目录>/color/right" "<数据目录>/prediction_no_augment"

# ━━━━ 阶段二：3D语义实例融合 ━━━━
cd PointCloud_Segement_v0/build
cmake .. && make
./Test_main ../test_data ../test_data/outputs/run01

# ━━━━ 阶段三 Step 1：点云投影 (C++) ━━━━
cd l2bim && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make pcd_projection && cd ..
./bin/pcd_projection <removed.pcd路径> output/run01

# ━━━━ 阶段三 Step 2：直线段检测 (Python) ━━━━
python examples/python/wall_regularization_v2.py \
    --input output/run01/projection_lines_refined.png \
    --output-dir output/run01
```

---

## 📚 文档索引

| 文档 | 内容 |
|------|------|
| **[environment.md](environment.md)** | 环境部署与依赖安装 |
| **[docs/](docs/)** | 技术路线详细文档 |
| **[knowledge/](knowledge/)** | 知识文档 |

---

## 🎯 创新贡献

| 创新点 | 层级 | 说明 |
|--------|------|------|
| **3D点集级IoU数据关联** | 核心创新 | 解决2D IoU视角敏感问题 |
| 体素哈希加速 | 工程优化 | 40倍加速视锥体筛选 |
| 贝叶斯多帧语义融合 | 方法创新 | 处理多帧语义不一致 |
| 语义先验过滤 | 工程优化 | CLIP过滤无关标签 |
| 关键帧选取 | 工程优化 | 减少冗余计算 |
| Z直方图自适应高度检测 | 工程优化 | 自动检测建筑高度 |

---

## 🔗 依赖的开源项目

- [RAM (Recognize Anything Model)](https://github.com/xinyu1205/recognize-anything)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [SAM (Segment Anything Model)](https://github.com/facebookresearch/segment-anything)
- [OpenAI CLIP](https://github.com/openai/CLIP)

---

## 📄 许可证

各子模块遵循其各自的开源许可证（详见子目录中的 LICENSE 文件）。
