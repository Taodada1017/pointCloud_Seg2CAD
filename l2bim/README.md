# L2BIM 项目说明（对外介绍版）

## 1. 项目简介
L2BIM 是一个面向建筑室内场景的点云处理项目，核心目标是从三维点云中提取可用于 BIM 对齐与结构分析的二维墙体表达。

当前常用工作链路分为两步：
1. C++ 投影阶段：将三维点云投影为二维栅格和二维墙体点云。
2. Python 检测/规则化阶段：对投影图中的墙线进行检测、合并与规则化。

## 2. 当前实际使用的核心程序
- 投影程序：`examples/cpp/pcd_projection.cpp`
- 检测程序：`examples/python/wall_regularization_v2.py`

项目中常用的数据与结果目录：
- 测试点云目录：`pcds/`
- 输出目录：`output/`

## 3. 处理流程（输入 -> 输出）
### 第一步：点云投影（C++）
输入：`.pcd` 或 `.ply` 点云文件（通常位于 `pcds/`）

主要处理：
- 地面与墙体分离
- 按高度带筛选墙体点
- 栅格化并生成投影图
- 可选线化后处理，导出线状二维墙体点云

典型输出（位于 `output/`）：
- `projection.png`：二维投影主图
- `projection_vote_counts.png`：多切片投票可视化
- `projection_lines_bin.png`：线二值图
- `projection_lines_thin.png`：细化后线图
- `projection_lines_refined.png`：线优化结果
- `projection_lines_overlay.png`：叠加可视化
- `walls_2d.pcd`：二维墙体点云（优先线状版本）
- `walls_2d_fill.pcd`：填充型二维点云备份
- `slices/`：分层投影中间结果

### 第二步：墙线检测与规则化（Python）
输入：投影阶段生成的二值/灰度墙体图（建议优先使用 `projection_lines_refined.png`）

主要处理：
- 曼哈顿方向自动对齐
- 规则/非规则区域分块判定
- Hough 线检测
- 共线段合并与正交拓扑修正
- 生成最终规则化墙线结果

典型输出（位于你指定的输出目录）：
- `v2_step0_raw_input.png`
- `v2_step0_input.png`
- `v2_step0_aligned_input.png`
- `v2_step1_region_mask.png`
- `v2_step2_regular_regions.png`
- `v2_step3_hough_lines.png`
- `v2_step5_original_lines.png`
- `v2_step6_extended_lines.png`
- `v2_step7_combined.png`
- `v2_step8_final_result.png`（最终结果）

## 4. 快速使用示例（Windows）
以下命令在项目根目录执行。

### 4.1 编译投影程序
```powershell
cmake -S . -B build_vs17_x64 -G "Visual Studio 17 2022" -A x64
cmake --build build_vs17_x64 --config Release --target pcd_projection
```

### 4.2 运行投影
```powershell
.\build_vs17_x64\Release\pcd_projection.exe .\pcds\your_test_cloud.pcd .\output
```

说明：
- 命令中的第 1 个参数是输入点云路径。
- 第 2 个参数是输出目录（建议使用 `output/`）。
- 若不追加更多参数，将使用程序内置默认参数。

### 4.3 运行墙线检测（Python）
```powershell
python .\examples\python\wall_regularization_v2.py --input .\output\projection_lines_refined.png --output-dir .\output
```

说明：
- 若你希望以原始投影作为输入，也可将 `--input` 换成 `./output/projection.png`。
- 脚本末尾会弹出可视化窗口；关闭窗口后进程结束。

## 5. 对外介绍时可使用的简版描述
L2BIM 通过“点云投影 + 墙线规则化”两阶段流程，将室内三维点云转化为稳定的二维墙体结构表达。第一阶段使用 C++ 程序进行墙体分离、栅格化与二维点云导出，第二阶段使用 Python 对投影结果进行几何规则化，最终输出结构清晰的墙线图和过程可追踪的中间结果，便于用于 BIM 对齐、结构分析与后续自动化处理。

## 6. 常见注意事项
- 输入点云建议已做基础去噪，且室内结构（墙体）占比足够高。
- 高度带参数（lowerZ/upperZ）会显著影响墙体投影质量，必要时应按数据集调参。
- 如果墙线过于碎片化，优先尝试 `projection_lines_refined.png` 作为检测输入。
- Python 依赖建议提前安装：`pip install -r requirements.txt`。
