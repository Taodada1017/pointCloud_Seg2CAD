# rma-dino-sam — 2D 开放词汇语义分割

> **阶段一**：利用 RAM + GroundingDINO + SAM 对室内多视角图像进行自动化开放词汇语义实例分割。

---

## 功能概述

本模块是 PointCloud2CAD Pipeline 的**第一阶段**，负责将 RGB 图像序列转换为像素级语义分割结果。处理流程：

```
输入图像 ──► RAM（自动标签生成）──► GroundingDINO（目标检测）──► SAM（实例分割）──► 输出 mask + label
```

1. **RAM**（Recognize Anything Model）：对每张图像自动生成语义标签文本（如 `chair`, `table`, `floor` 等）
2. **GroundingDINO**：根据 RAM 输出的文本标签进行开放词汇目标检测，输出检测框和置信度
3. **SAM**（Segment Anything Model）：根据检测框生成像素级实例分割掩码
4. 最终输出每张图像的 `_mask.png`（16 位 PNG）和 `_label.json`

---

## 目录结构

```
rma-dino-sam/
├── main.py                     # ★ 主入口脚本，集成 RAM+DINO+SAM 全流程
├── README.md                   # 本文件
├── clip/                       # OpenAI CLIP 模型（分词器）
│   └── clip/                   # 核心模块 (clip.py, model.py, simple_tokenizer.py)
├── recognize-anything/         # RAM 图像标签识别模型
│   ├── ram/                    # 模型代码 + 标签列表 (4000+ 中英文标签)
│   ├── inference_ram.py        # RAM 推理入口
│   └── requirements.txt
├── GroundingDINO/              # Grounding DINO 开放集目标检测
│   ├── groundingdino/          # 模型代码 (config, models, util)
│   ├── requirements.txt
│   └── setup.py
├── segment_anything/           # SAM 分割一切模型
│   ├── segment_anything/       # 模型代码 (modeling, predictor)
│   └── setup.py
└── EfficientSAM/               # 高效 SAM 变体（可选，未在主流程使用）
    ├── EdgeSAM/
    ├── MobileSAM/
    ├── LightHQSAM/
    ├── FastSAM/
    └── RepViTSAM/
```

---

## 环境配置

> 详细的跨设备部署流程请参阅项目根目录下的 [`INSTALL.md`](../INSTALL.md)。

### 快速开始

**推荐环境**：Python 3.11 + PyTorch ≥ 2.6（CPU 或 GPU）

```bash
# 1. 创建 conda 环境
conda create -n gsam python=3.11
conda activate gsam

# 2. 安装 PyTorch（参考 https://pytorch.org/get-started/previous-versions/）
pip3 install torch torchvision

# 3. 安装子模块（在 rma-dino-sam/ 根目录下执行）
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
cd recognize-anything && pip install -r requirements.txt && pip install -e . && cd ..

# 4. 安装 CLIP（如 pip 安装 RAM 时 git 连接失败）
cd clip && pip install ftfy regex tqdm && pip install . && cd ..

# 5. 其他依赖
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

> ⚠️ **注意**：RAM 的 `requirements.txt` 最后一行需要从 GitHub 下载 CLIP，如果网络连接失败，需手动安装本地 `clip/` 目录中的 CLIP 源码。

---

## 模型权重下载

| 模型 | 文件名 | 下载地址 |
|------|--------|---------|
| GroundingDINO (SwinT) | `groundingdino_swint_ogc.pth` | [GitHub Release](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) |
| SAM (ViT-H) | `sam_vit_h_4b8939.pth` | [Meta AI](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| RAM (14M) | `ram_swin_large_14m.pth` | [HuggingFace](https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/ram_swin_large_14m.pth) |
| bert-base-uncased（分词器） | `bert-base-uncased/` | [夸克网盘](https://pan.quark.cn/s/e6cda95e1ca1) |

> 三个模型的 checkpoint 合集：[夸克网盘](https://pan.quark.cn/s/2fd76d205f7a)

> ⚠️ **仅推荐 RAM (14M) 版本**，使用 RAM++ 会出现兼容性问题。

---

## 本地配置（必须）

### 1. 配置 bert-base-uncased 路径

模型推理需要本地的 `bert-base-uncased` 分词器，否则代码会尝试联网下载并报 443 错误。

**DINO 配置** — 修改 `GroundingDINO/groundingdino/util/get_tokenlizer.py`：

```python
# 在文件顶部添加全局变量
local_path = "/你的路径/bert-base-uncased"  # 使用绝对路径

# 修改 get_tokenlizer 函数中的：
tokenizer = AutoTokenizer.from_pretrained(local_path)
```

**RAM 配置** — 修改 `recognize-anything/ram/models/utils.py` 第 130 行：

```python
tokenizer = BertTokenizer.from_pretrained("/你的路径/bert-base-uncased")
```

### 2. 配置模型权重路径

修改 `main.py` 中的以下变量为本地权重文件的绝对路径：

```python
ram_checkpoint = "/你的路径/ram_swin_large_14m.pth"
grounded_checkpoint = "/你的路径/groundingdino_swint_ogc.pth"
sam_checkpoint = "/你的路径/sam_vit_h_4b8939.pth"
```

### 3. 已知问题修复

**transformers 兼容性**：如果遇到 `ImportError: cannot import name 'apply_chunking_to_forward'`，请参考 [此解决方案](https://blog.csdn.net/m0_75101512/article/details/153682726) 修改 `recognize-anything/ram/models/bert.py` 第 28 行。

---

## 运行方式

```bash
cd rma-dino-sam
python main.py "<输入图像文件夹>" "<输出文件夹>"
```

- **输入**：文件夹内的图像文件（支持 `.jpg` / `.jpeg` / `.png` / `.bmp` / `.tiff`）
- **输出**：

| 文件 | 格式 | 说明 |
|------|------|------|
| `{name}_mask.png` | 16 位 PNG | 像素值为实例 ID（0 = 背景） |
| `{name}_label.json` | JSON | 包含 `raw_tags`、`tags`、每个 mask 的标签/置信度/检测框 |

### label.json 格式示例

```json
{
  "raw_tags": "building. chair. floor. table...",
  "tags": "chair. floor. table",
  "mask": [
    {"value": 0, "label": "background"},
    {"value": 1, "labels": {"chair": 0.57}, "box": [159.7, 816.5, 341.4, 1013.9]},
    {"value": 2, "labels": {"floor": 0.46}, "box": [772.2, 843.6, 1598.1, 1596.8]}
  ]
}
```

---

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `box_threshold` | 0.3 | 检测框置信度阈值 |
| `text_threshold` | 0.25 | 文本匹配阈值 |
| `iou_threshold` | 0.5 | NMS 重叠阈值（防止重复检测） |
| `sam_version` | `vit_h` | SAM 模型版本 |
| `device` | `cpu` | 推理设备（GPU 需 ≥16GB 显存） |

---

## 参考资料

- **源项目**：[Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- **本地部署参考**：[CSDN 教程](https://blog.csdn.net/weixin_44362044/article/details/136149372)
