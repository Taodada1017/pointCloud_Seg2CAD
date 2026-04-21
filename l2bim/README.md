# l2bim — 室内点云 → 2D CAD 直线段提取

## 1. 项目简介

l2bim 是 PointCloud2CAD 流水线的**阶段三**，负责将三维室内点云转化为 **2D CAD 平面直线段**。

核心功能：从已去除家具/物体的室内点云（通常来自阶段二的 `removed.pcd`）中，通过"**点云投影 + 直线段检测与规则化**"两步流程，提取结构清晰的二维墙体直线段。

> **说明**：本项目源自 LiDAR2BIM 研究项目，代码中保留了部分 BIM 配准相关的历史代码（`backend/`、`frontend/feature/`、`bim.cpp`、`submap.cpp` 等），这些模块与当前 CAD 直线段提取功能**无关**，不参与编译和运行。当前实际使用的仅为投影和线段检测相关模块。

### 在整体流水线中的位置

```
阶段一 (rma-dino-sam)        阶段二 (PointCloud_Segement_v0)       阶段三 (l2bim)
RGB 图像 → 2D 语义分割  →  3D 语义实例融合 → removed.pcd  →  点云投影 → 直线段检测/规则化 → CAD 平面图
                                                                 ↑                              ↓
                                                          pcd_projection.cpp        wall_regularization_v2.py
```

### 输入来源

| 来源 | 说明 |
|------|------|
| `removed.pcd`（阶段二输出） | 从原始 SLAM 点云中**剔除所有已识别家具/物体后**保留的建筑结构点云（墙壁 + 地板 + 天花板），是最理想的输入 |
| 独立 `.pcd` / `.ply` 文件 | 也可直接用任意室内点云运行（如 `pcds/` 目录下的测试数据），无需依赖前置阶段 |

### 两步处理流程

| 步骤 | 程序 | 语言 | 功能 |
|------|------|------|------|
| Step 1 | `examples/cpp/pcd_projection.cpp` | C++ | RANSAC 地面/墙体分离 → 坐标对齐 → 多切片投票投影 → 骨架化 → Hough 线段检测与合并 |
| Step 2 | `examples/python/wall_regularization_v2.py` | Python | 曼哈顿方向对齐 → 规则/非规则区域分类 → Hough 线检测 → 共线合并 → 正交拓扑修正 |

---

## 2. 项目目录结构

```
l2bim/
├── CMakeLists.txt                  # CMake 构建配置（构建目标：pcd_projection 等）
├── README.md                       # 本文档
├── LICENSE
├── requirements.txt                # Python 依赖
│
├── examples/                       # ★ 核心可执行程序 & 脚本
│   ├── cpp/                        #   C++ 主程序
│   │   ├── pcd_projection.cpp      #   ★ 主投影程序（Step 1，当前使用）
│   │   ├── pcd_projection_before.cpp  # pcd_projection 的历史版本
│   │   ├── pcd2line_test.cpp       #   完整管线测试（load→RANSAC→PCD2d→raster→ED→Hough→merge）
│   │   ├── projection_2d.cpp       #   密度投影替代方案（以 walls_2d_fill.pcd 为输入）
│   │   ├── img_plus.cpp            #   投影图形态学后处理工具（闭运算 + 连通域过滤）
│   │   ├── demo_reg.cpp            #   [历史代码] BIM 配准演示（当前不使用）
│   │   └── reg_bm.cpp              #   [历史代码] BIM 配准基准测试（当前不使用）
│   ├── python/
│   │   ├── wall_regularization_v2.py      # ★ 直线段规则化主程序（Step 2，当前使用）
│   │   ├── wall_regularization_v2_fix.py  # 规则化修改版（跳过区域分类，不做共线合并）
│   │   └── data_process/           #   [历史代码] BIM 配准数据预处理
│   └── scripts/
│       └── benchmark.sh            #   [历史代码] BIM 配准批量测试脚本
│
├── src/                            # C++ 源码库
│   ├── frontend/                   #   前端处理模块
│   │   ├── geometry/               #     几何处理
│   │   │   ├── point_cloud.cpp     #     ★ PCD2d 类（地面拟合、墙体裁剪、投影压平）
│   │   │   ├── lineseg.cpp         #     ★ 线段表示与操作
│   │   │   ├── corner.cpp          #     ★ 角点检测（线段交叉点）
│   │   │   ├── ED.cpp              #     ★ Edge Drawing 边缘检测
│   │   │   ├── EDColor.cpp         #     ★ 彩色边缘检测
│   │   │   ├── EDLines.cpp         #     ★ ED 线段检测
│   │   │   ├── NFA.cpp             #     ★ NFA 统计验证
│   │   │   ├── largepcd.cpp        #     大规模点云处理
│   │   │   ├── merge_projection_images.cpp  # 投影图合并
│   │   │   ├── bim.cpp             #     [历史代码] BIM 模型加载
│   │   │   └── submap.cpp          #     [历史代码] LiDAR 子图管理
│   │   └── feature/                #     [历史代码] BIM 配准特征模块
│   │       ├── descriptor.cpp      #     几何三元组描述子（配准用）
│   │       └── match.cpp           #     特征匹配（配准用）
│   ├── backend/                    #   [历史代码] BIM 配准后端
│   │   ├── hough/                  #     Hough 投票配准
│   │   │   ├── hough_voting.cpp    #     (x,y,yaw) 位姿空间投票
│   │   │   ├── pose_2d.cpp         #     2D 位姿数据结构
│   │   │   └── verification.cpp    #     配准位姿验证
│   │   └── reglib.cpp              #     全局配准核心
│   └── utils/                      #   工具模块
│       ├── raster_utils.cpp        #     ★ PointRaster/LineRaster 栅格化、形态学运算
│       ├── cfg.cpp                 #     配置读取
│       ├── file_io.cpp             #     文件 IO
│       ├── utils.cpp               #     通用工具函数
│       ├── analysis.cpp            #     [历史代码] 配准分析
│       └── clustering_utils.cpp    #     [历史代码] 聚类（配准用）
│
├── include/                        # C++ 头文件（与 src/ 结构对应）
│   ├── frontend/geometry/          #   point_cloud.h, lineseg.h, corner.h, ED*.h 等
│   ├── frontend/feature/           #   [历史代码] descriptor.h, match.h
│   ├── backend/                    #   [历史代码] reglib.h, hough/*.h
│   ├── utils/                      #   raster_utils.h, cfg.h 等
│   └── global_definition/          #   全局定义
│
├── pcds/                           # ★ 测试点云数据
│                                   #   001.pcd ~ 008.pcd, 3.3.pcd 等
│
├── output/                         # ★ 输出结果目录
│   ├── detect/                     #   线检测结果（按数据集编号分子目录）
│   └── projection/                 #   投影结果（按数据集编号分子目录）
│
├── configs/                        # YAML 配置文件
│   ├── interval/                   #   [历史代码] 配准参数配置（17 个 .yaml）
│   └── pointsam/                   #   Point-SAM 相关配置（38 个 .yaml）
│
├── preprocess/                     # Python 预处理工具集
│   ├── datasets/                   #   数据集管理（含 .pcd 与 .png 中间产物）
│   ├── geometry/                   #   几何计算模块
│   ├── cfg_loader.py               #   配置加载器
│   ├── clustering_utils.py         #   聚类工具
│   ├── config.py                   #   配置定义
│   └── utils.py                    #   通用工具
│
├── docs/                           # 文档与示意图
│   ├── overview.png, pipeline.png  #   项目总览/流程图
│   ├── demo*.png                   #   效果演示图
│   ├── install.md                  #   安装说明
│   ├── prepare.md                  #   数据准备说明
│   ├── demo.md                     #   演示说明
│   └── benchmark.md                #   基准测试说明
│
├── Thirdparty/                     # 第三方库
│   ├── backward-cpp/               #   堆栈追踪（调试用）
│   ├── nanoflann/                  #   高效 KD-Tree 最近邻搜索
│   └── point-sam/                  #   Point-SAM 模型相关
│
├── cmake/                          # CMake 查找模块（FindXXX.cmake）
│
├── api.py                          # API 接口（Web 服务入口）
├── oliver_preview.py               # 预览可视化脚本
├── reflect_filter_single_pcd.py    # 反射率过滤单个 PCD 文件
├── transfer_npy.py                 # NumPy 格式转换工具
├── transfer_txt.py                 # 文本格式转换工具
│
├── bin/                            # Windows 预编译二进制 & DLL
├── build_projection/               # CMake 构建目录（pcd_projection 单独构建）
├── build_projection2d_test/        # CMake 构建目录（projection_2d 单独构建）
├── build_vs17_x64/                 # CMake 构建目录（完整 VS2022 构建）
└── build/                          # CMake 构建目录（通用构建）
```

### 当前使用 vs 历史代码

| 模块 | 用途 | 状态 |
|------|------|------|
| `pcd_projection.cpp` | 点云 → 2D 投影图 + Hough 线段 | **★ 当前使用** |
| `wall_regularization_v2.py` | 投影图 → 规则化 CAD 直线段 | **★ 当前使用** |
| `src/frontend/geometry/point_cloud.cpp` | PCD2d 投影核心类 | **★ 当前使用**（被 pcd_projection 编译链接） |
| `src/utils/raster_utils.cpp` | 栅格化工具 | **★ 当前使用**（被 pcd_projection 编译链接） |
| `src/frontend/geometry/ED*.cpp, NFA.cpp, lineseg.cpp, corner.cpp` | ED 线检测 + 线段/角点管理 | **★ 被 pcd2line_test 使用** |
| `src/backend/` 整个目录 | BIM 全局配准（Hough 位姿投票 + 验证） | 历史代码，不使用 |
| `src/frontend/feature/` 整个目录 | BIM 配准特征描述子 + 匹配 | 历史代码，不使用 |
| `src/frontend/geometry/bim.cpp` | BIM 模型加载管理 | 历史代码，不使用 |
| `src/frontend/geometry/submap.cpp` | LiDAR 子图加载管理 | 历史代码，不使用 |
| `examples/cpp/demo_reg.cpp, reg_bm.cpp` | BIM 配准 demo/benchmark | 历史代码，不使用 |

---

## 3. 核心依赖

### C++ 依赖（CMake 管理）

仅 `pcd_projection` 构建目标实际需要的依赖：

| 库 | 用途 |
|---|---|
| **PCL** (Point Cloud Library) | 点云 IO、RANSAC、法向量估计、体素滤波 |
| **OpenCV** | 图像处理（骨架化、Hough 变换、形态学运算） |
| **Eigen** | 矩阵运算、坐标变换 |

> CMakeLists.txt 中还声明了 yaml-cpp、glog、OpenMP、nanoflann、point-sam 等依赖，这些主要供完整静态库（历史 BIM 配准模块）使用。`pcd_projection` 目标仅链接 PCL + OpenCV。

### Python 依赖（`requirements.txt`）

```
opencv-python-headless==4.10.0.84
numpy==1.26.4
```

> `wall_regularization_v2.py` 仅使用 OpenCV + NumPy。`requirements.txt` 中列出的 open3d、scipy、shapely 等是其他预处理脚本使用的，主流程不需要。

安装：`pip install -r requirements.txt`

---

## 4. 处理流程详解

### Step 1：点云投影（C++ `pcd_projection`）

**入口**：`examples/cpp/pcd_projection.cpp`

**用法**：
```bash
pcd_projection <input.pcd> <output_dir> [gridSize] [resolutionScale] [lowerZ] [upperZ] [voxel] [numSlices] [minVotes] [houghThreshold] [minLineLength] [maxLineGap] [mergeAngleDeg] [mergeDistPx]
```

**处理步骤**：
1. `loadCloud()` — 加载 `.pcd` 或 `.ply` 点云
2. `splitGroundWalls()` — RANSAC 分离地面，利用垂直法向约束提取墙体
3. `alignWallsToXYZ()` — 利用地面法向量对齐坐标系 + yaw 角对齐
4. 多切片投票（`numSlices` 层高度切片叠加）
5. `rasterizePointsToCounts32F_AA()` — 栅格化生成 2D 投影图
6. `makeLineBinaryFromProjection()` — 密度阈值二值化
7. `skeletonizeBinary()` — 形态学骨架化
8. `redrawMergedHoughLines()` — Hough 线段检测 + 合并
9. `maskToPointCloud2D()` — 导出二维点云

**输出文件**：
| 文件 | 说明 |
|------|------|
| `projection.png` | 二维投影主图 |
| `projection_vote_counts.png` | 多切片投票热力图 |
| `projection_lines_bin.png` | 线二值图 |
| `projection_lines_thin.png` | 骨架化细线图 |
| `projection_lines_refined.png` | Hough 优化后线图（**推荐作为 Step 2 输入**） |
| `projection_lines_overlay.png` | 叠加可视化 |
| `walls_2d.pcd` | 二维墙体点云（线状版本） |
| `walls_2d_fill.pcd` | 二维墙体点云（填充版本） |
| `slices/` | 各高度切片中间结果 |

### Step 2：直线段检测与规则化（Python `wall_regularization_v2`）

**入口**：`examples/python/wall_regularization_v2.py`

**用法**：
```bash
python wall_regularization_v2.py --input <projection_image> --output-dir <output_dir>
```

**处理步骤**：
1. 加载灰度投影图
2. **曼哈顿方向自动对齐** — 统计所有前景像素方向直方图，旋转图像使主墙方向水平
3. **规则/非规则区域分类** — 将图像分为 30×30 网格，用 `cv2.fitLine` 判定每个块是否可拟合为直线
4. **Hough 线段检测** — 在规则区域上执行 `HoughLinesP`，提取直线段
5. **共线线段合并** — 按角度聚类（水平/垂直），判断共线性 + 路径连通性验证，合并碎片线段
6. **正交拓扑修正** — 检测近垂直相交线段对，将端点吸附到交叉点，保证墙角闭合
7. 生成最终规则化直线段结果

**输出文件**（10 张过程图）：
| 文件 | 说明 |
|------|------|
| `v2_step0_raw_input.png` | 原始输入 |
| `v2_step0_input.png` | 预处理后输入 |
| `v2_step0_aligned_input.png` | 曼哈顿对齐后 |
| `v2_step1_region_mask.png` | 区域分类掩码（绿=规则，红=非规则） |
| `v2_step2_regular_regions.png` | 规则区域提取 |
| `v2_step3_hough_lines.png` | Hough 线段检测结果 |
| `v2_step5_original_lines.png` | 原始检测线段 |
| `v2_step6_extended_lines.png` | 共线合并后的延伸线段 |
| `v2_step7_combined.png` | 对比图（红=原始，绿=扩展） |
| `v2_step8_final_result.png` | **★ 最终 CAD 直线段结果** |

---

## 5. 快速开始

### 5.1 编译 C++ 投影程序

**macOS / Linux**：
```bash
cd l2bim
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make pcd_projection -j$(nproc)
```

**Windows（Visual Studio 2022）**：
```powershell
cd l2bim
cmake -S . -B build_vs17_x64 -G "Visual Studio 17 2022" -A x64
cmake --build build_vs17_x64 --config Release --target pcd_projection
```

> 只需编译 `pcd_projection` 目标。该目标仅编译 `pcd_projection.cpp` + `point_cloud.cpp` + `raster_utils.cpp`，不依赖 BIM 配准模块。

### 5.2 运行 Step 1（投影）

```bash
# macOS / Linux
./build/pcd_projection ./pcds/001.pcd ./output

# Windows
.\build_vs17_x64\Release\pcd_projection.exe .\pcds\001.pcd .\output
```

- 第 1 个参数：输入点云路径
- 第 2 个参数：输出目录
- 不追加额外参数则使用程序内置默认值

### 5.3 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 5.4 运行 Step 2（规则化）

```bash
python examples/python/wall_regularization_v2.py \
  --input ./output/projection_lines_refined.png \
  --output-dir ./output
```

- 建议使用 `projection_lines_refined.png` 作为输入（Hough 优化后效果更好）
- 也可尝试 `projection.png` 作为输入

---

## 6. 其他可执行程序

### 当前使用的构建目标

| 构建目标 | 源文件 | 说明 |
|---|---|---|
| `pcd_projection` | `pcd_projection.cpp` | ★ 主投影程序（仅链接 PCL + OpenCV） |
| `pcd2line_test` | `pcd2line_test.cpp` | 完整管线测试（PCD2d + ED 线检测 + Hough + 角点） |
| `projection_2d` | `projection_2d.cpp` | 密度阈值投影替代方案（输入 `walls_2d_fill.pcd`） |
| `img_plus` | `img_plus.cpp` | 投影图形态学后处理（闭运算 + 连通域过滤） |

### Python 替代版本

| 脚本 | 说明 |
|---|---|
| `wall_regularization_v2_fix.py` | 规则化修改版：跳过区域分类（全部当作规则区域），不做共线合并/拓扑延伸 |

### 历史代码构建目标（BIM 配准，当前不使用）

| 构建目标 | 源文件 | 说明 |
|---|---|---|
| `demo_reg` | `demo_reg.cpp` | BIM 配准单次演示 |
| `reg_bm` | `reg_bm.cpp` | BIM 配准批量 benchmark |

---

## 7. 注意事项

- 输入点云建议已做基础去噪，且室内结构（墙体）占比足够高。
- 高度带参数（`lowerZ` / `upperZ`）会显著影响墙体投影质量，必要时应按数据集调参。
- 如果墙线过于碎片化，优先尝试 `projection_lines_refined.png` 作为 Step 2 输入。
- C++ 部分使用 **PCL**（Point Cloud Library），与阶段二使用的 **Open3D** 不同，两者点云格式（`.pcd`）兼容。
- Windows 预编译产物在 `bin/` 目录下，macOS/Linux 需自行编译。
- `pcd_projection` 目标编译轻量，不依赖 BIM 配准模块（`backend/`、`feature/` 等），可快速独立构建。
