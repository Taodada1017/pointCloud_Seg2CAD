#!/usr/bin/env python3
"""
阶段二 → 阶段三 转换脚本
=========================
将 PointCloud_Segement_v0 (阶段二) 的输出转换为 2D 线段。

阶段二输出:
  - colorized.pcd    : 原始 SLAM 点云 (带颜色)
  - removed.pcd      : 剔除家具后的背景点云 (墙壁+地板+天花板)
  - info.txt         : 实例语义标签 (文本)
  - points.txt       : 实例点云下标 (二进制)

本脚本做三件事:
  1. 从 removed.pcd 中按语义标签分离出 walls.pcd 和 ground.pcd
  2. 将墙壁点云投影为 2D 图像
  3. 用 Hough 变换检测 2D 线段并输出

用法:
  python stage2_to_stage3.py <stage2_output_dir> [--slam <slam_pcd_path>] [--output <output_dir>]

示例:
  python stage2_to_stage3.py PointCloud_Segement_v0/test_data/outputs/obb-merge02 \
      --slam PointCloud_Segement_v0/test_data/colorized.pcd \
      --output pipeline_output
"""

import os
import sys
import struct
import argparse
import numpy as np

# 将阶段三的代码路径加入 sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE3_DIR = os.path.join(SCRIPT_DIR, "LiDAR2BIM-Registration")
if STAGE3_DIR not in sys.path:
    sys.path.insert(0, STAGE3_DIR)

try:
    import open3d as o3d
    import cv2
    from PIL import Image
except ImportError as e:
    print(f"[错误] 缺少依赖: {e}")
    print("请安装: pip install open3d opencv-python pillow numpy")
    sys.exit(1)


# ============================================================
# 第一部分: 解析阶段二的输出文件
# ============================================================

def read_info_txt(info_path: str) -> dict:
    """
    解析 info.txt，返回每个实例的语义标签信息。
    
    格式: <instance_id> <frame_id> <label1>:<score1>,<label2>:<score2>,...
    
    返回: {instance_id: {"frame_id": int, "labels": {label: score}, "predicted": (label, score)}}
    """
    instances = {}
    with open(info_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                print(f"  [警告] info.txt 行格式异常，跳过: {line}")
                continue
            
            instance_id = int(parts[0])
            frame_id = int(parts[1])
            labels_str = parts[2]
            
            labels = {}
            best_label = ("unknown", 0.0)
            for pair in labels_str.split(","):
                if ":" not in pair:
                    continue
                label, score_str = pair.rsplit(":", 1)
                try:
                    score = float(score_str)
                except ValueError:
                    continue
                labels[label] = score
                if score > best_label[1]:
                    best_label = (label, score)
            
            instances[instance_id] = {
                "frame_id": frame_id,
                "labels": labels,
                "predicted": best_label,
            }
    return instances


def read_points_bin(points_path: str) -> dict:
    """
    解析 points.txt (二进制格式)，返回每个实例的点云下标。
    
    二进制格式:
      [uint32_t instance_id][size_t count][size_t[count] indices]
      重复直到 EOF
    
    返回: {instance_id: np.ndarray of uint64 indices}
    """
    instances = {}
    with open(points_path, "rb") as f:
        while True:
            # 读取 instance_id (uint32_t, 4 bytes, little-endian)
            data = f.read(4)
            if len(data) < 4:
                break
            instance_id = struct.unpack("<I", data)[0]
            
            # 读取 indices_count (size_t = uint64, 8 bytes)
            data = f.read(8)
            if len(data) < 8:
                print(f"  [警告] points.txt 中实例 {instance_id} 的 count 不完整")
                break
            indices_count = struct.unpack("<Q", data)[0]
            
            # 合理性检查
            if indices_count > 500_000_000:  # 5亿
                print(f"  [警告] 实例 {instance_id} 的下标数量异常: {indices_count}")
                break
            
            # 读取 indices 数组 (size_t[] = uint64[])
            if indices_count > 0:
                data = f.read(indices_count * 8)
                if len(data) < indices_count * 8:
                    print(f"  [警告] 实例 {instance_id} 的下标数据不完整")
                    break
                indices = np.frombuffer(data, dtype=np.uint64)
            else:
                indices = np.array([], dtype=np.uint64)
            
            instances[instance_id] = indices
    return instances


# ============================================================
# 第二部分: 语义分离 - 从混合点云中分出 walls 和 ground
# ============================================================

# 地面类别: 阶段二的 remove_all_instances 保留了 floor、ceiling、carpet
GROUND_LABELS = {"floor", "carpet"}
CEILING_LABELS = {"ceiling"}
# 其余所有点视为 "墙壁" (结构背景)

def separate_point_cloud(
    slam_pcd_path: str,
    info_path: str,
    points_path: str,
    output_dir: str,
) -> tuple:
    """
    从阶段二输出中分离墙壁点云和地面点云。
    
    策略:
      1. 读取原始 SLAM 点云 (colorized.pcd)
      2. 读取 info.txt 获取每个实例的语义标签
      3. 读取 points.txt 获取每个实例对应的点云下标
      4. 标记 floor/carpet 实例的点为 "地面"
      5. 标记 ceiling 实例的点为 "天花板" (排除)
      6. 标记其他非结构实例的点为 "家具" (排除)
      7. 原始点云中剩余未标记的点 = "墙壁/结构背景"
    
    返回: (walls_pcd, ground_pcd) - 两个 Open3D PointCloud 对象
    """
    print("\n[步骤1] 读取原始 SLAM 点云...")
    slam_pcd = o3d.io.read_point_cloud(slam_pcd_path)
    slam_points = np.asarray(slam_pcd.points)
    slam_colors = np.asarray(slam_pcd.colors) if slam_pcd.has_colors() else None
    N = len(slam_points)
    print(f"  SLAM 点云: {N} 个点, 有颜色: {slam_colors is not None}")
    
    print("\n[步骤2] 读取实例语义信息 (info.txt)...")
    info = read_info_txt(info_path)
    print(f"  共 {len(info)} 个实例")
    
    # 统计语义分布
    label_counts = {}
    for inst_id, inst_info in info.items():
        label = inst_info["predicted"][0]
        label_counts[label] = label_counts.get(label, 0) + 1
    print("  语义标签分布:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {label}: {count} 个实例")
    
    print("\n[步骤3] 读取实例点云下标 (points.txt)...")
    point_indices = read_points_bin(points_path)
    print(f"  共 {len(point_indices)} 个实例有点云数据")
    
    print("\n[步骤4] 按语义分离点云...")
    # 创建标记数组: 0=未标记(墙壁), 1=地面, 2=天花板, 3=家具(要剔除)
    labels_array = np.zeros(N, dtype=np.int8)
    
    ground_count = 0
    ceiling_count = 0
    furniture_count = 0
    
    for inst_id, indices in point_indices.items():
        if inst_id not in info:
            # 没有语义信息的实例，视为家具剔除
            valid_mask = indices < N
            valid_indices = indices[valid_mask].astype(np.int64)
            labels_array[valid_indices] = 3
            furniture_count += len(valid_indices)
            continue
        
        predicted_label = info[inst_id]["predicted"][0]
        valid_mask = indices < N
        valid_indices = indices[valid_mask].astype(np.int64)
        
        if predicted_label in GROUND_LABELS:
            labels_array[valid_indices] = 1
            ground_count += len(valid_indices)
        elif predicted_label in CEILING_LABELS:
            labels_array[valid_indices] = 2
            ceiling_count += len(valid_indices)
        else:
            # 其他物体实例（家具等），要剔除
            labels_array[valid_indices] = 3
            furniture_count += len(valid_indices)
    
    wall_mask = (labels_array == 0)
    ground_mask = (labels_array == 1)
    
    wall_count = np.sum(wall_mask)
    
    print(f"  墙壁/背景: {wall_count} 点")
    print(f"  地面: {ground_count} 点")
    print(f"  天花板: {ceiling_count} 点")
    print(f"  家具(剔除): {furniture_count} 点")
    
    # 构建墙壁点云
    walls_pcd = o3d.geometry.PointCloud()
    walls_pcd.points = o3d.utility.Vector3dVector(slam_points[wall_mask])
    if slam_colors is not None:
        walls_pcd.colors = o3d.utility.Vector3dVector(slam_colors[wall_mask])
    
    # 构建地面点云
    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(slam_points[ground_mask])
    if slam_colors is not None:
        ground_pcd.colors = o3d.utility.Vector3dVector(slam_colors[ground_mask])
    
    # 保存分离后的点云
    os.makedirs(output_dir, exist_ok=True)
    walls_path = os.path.join(output_dir, "walls.pcd")
    ground_path = os.path.join(output_dir, "ground.pcd")
    
    o3d.io.write_point_cloud(walls_path, walls_pcd)
    o3d.io.write_point_cloud(ground_path, ground_pcd)
    print(f"\n  已保存: {walls_path} ({wall_count} 点)")
    print(f"  已保存: {ground_path} ({ground_count} 点)")
    
    return walls_pcd, ground_pcd


# ============================================================
# 第三部分: 点云 → 2D 线段 (直接使用 Points2Image + Hough)
# ============================================================

def points2image(points: np.ndarray, scale: int = 60, dilate_kernel: int = 3):
    """
    将 Nx3 点云投影到 z=0 平面并生成二值图像。
    
    直接复用 LiDAR2BIM-Registration/preprocess/geometry/pc2img.py 的逻辑，
    但不依赖其 import 链。
    
    返回: (pil_image, T_hw2xy) - PIL 图像和坐标变换矩阵
    """
    xyz = points.copy()
    xyz[:, 2] = 0  # 投影到 z=0
    
    W, H = np.max(xyz, axis=0)[:2].astype(int) * scale
    img_size = (H + 1, W + 1)
    
    # xyz → 像素坐标 (hw)
    T_xyz2hw = np.array([[scale, 0, 0, 0], [0, -scale, 0, H], [0, 0, 0, 1]])
    T_hw2xy = np.array([[1/scale, 0, 0], [0, -1/scale, H/scale], [0, 0, 1]])
    
    xyz1 = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    hw = np.dot(T_xyz2hw, xyz1.T).T[:, :2]
    
    # 生成二值图像
    hw = np.round(hw).astype(int)
    valid = (hw[:, 0] >= 0) & (hw[:, 0] < img_size[1]) & (hw[:, 1] >= 0) & (hw[:, 1] < img_size[0])
    hw = hw[valid]
    
    hw_mat = np.zeros(img_size, dtype=np.uint8)
    hw_mat[hw[:, 1], hw[:, 0]] = 255
    
    # 膨胀
    kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    hw_mat = cv2.dilate(hw_mat, kernel, iterations=1)
    
    pil_img = Image.fromarray(255 - hw_mat)
    return pil_img, T_hw2xy


def hough_detect(img_array: np.ndarray, threshold=60, min_line_length=30, max_line_gap=30):
    """
    在二值图像上用 HoughLinesP 检测线段。
    
    返回: Nx4 numpy 数组, 每行 [x1, y1, x2, y2] (像素坐标)
    """
    lines = cv2.HoughLinesP(
        img_array,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return np.array([]).reshape(0, 4)
    return lines.reshape(-1, 4)


def pcd_to_2d_linesegs(
    walls_pcd: o3d.geometry.PointCloud,
    output_dir: str,
    min_height: float = 0.2,
    max_height: float = 1.9,
    scale: int = 60,
    dilate_kernel: int = 3,
    hough_threshold: int = 60,
    hough_min_length: int = 30,
    hough_max_gap: int = 30,
    voxel_size: float = 0.01,
):
    """
    墙壁点云 → 2D 线段。
    
    步骤:
      1. 按高度裁剪 (只保留 min_height ~ max_height 之间的点)
      2. 体素下采样
      3. 投影到 z=0 平面
      4. 平移使坐标为正
      5. 生成二值图像 (Points2Image)
      6. Hough 线段检测
      7. 像素坐标 → 物理坐标变换
    
    返回: Nx4 numpy 数组, 每行 [x1, y1, x2, y2] (物理坐标, 单位: 米)
    """
    points = np.asarray(walls_pcd.points)
    print(f"\n[线段提取] 输入墙壁点云: {len(points)} 点")
    
    # 1. 按高度裁剪
    height_mask = (points[:, 2] > min_height) & (points[:, 2] < max_height)
    points = points[height_mask]
    print(f"  高度裁剪 [{min_height}, {max_height}]: {len(points)} 点")
    
    if len(points) < 10:
        print("  [错误] 裁剪后点太少，无法生成线段")
        return np.array([]).reshape(0, 4)
    
    # 2. 体素下采样
    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(points)
    cropped_pcd = cropped_pcd.voxel_down_sample(voxel_size=voxel_size)
    points = np.asarray(cropped_pcd.points)
    print(f"  体素下采样 (voxel={voxel_size}): {len(points)} 点")
    
    # 3. 投影到 z=0
    points[:, 2] = 0
    
    # 4. 平移使坐标为正 (Points2Image 要求)
    xy_min = np.min(points[:, :2], axis=0)
    points[:, 0] -= xy_min[0]
    points[:, 1] -= xy_min[1]
    
    # 补偿一个小 epsilon 避免坐标为 0
    points[:, 0] += 0.01
    points[:, 1] += 0.01
    
    # 5. 生成图像
    print(f"  生成二值图像 (scale={scale}, dilate={dilate_kernel})...")
    pil_img, T_hw2xy = points2image(points, scale=scale, dilate_kernel=dilate_kernel)
    
    # 保存中间图像
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, "walls_2d.png")
    pil_img.save(img_path)
    print(f"  已保存 2D 投影图: {img_path}")
    
    # 6. Hough 线段检测
    img_array = np.array(pil_img)
    img_inv = 255 - img_array  # 反转
    
    lines_px = hough_detect(
        img_inv,
        threshold=hough_threshold,
        min_line_length=hough_min_length,
        max_line_gap=hough_max_gap,
    )
    print(f"  Hough 检测到 {len(lines_px)} 条线段 (像素坐标)")
    
    if len(lines_px) == 0:
        return np.array([]).reshape(0, 4)
    
    # 7. 像素坐标 → 物理坐标
    # T_hw2xy: 3x3 矩阵, 将 [h, w, 1] → [x, y, 1]
    rot = T_hw2xy[:2, :2]   # 2x2
    t = T_hw2xy[:2, 2]      # 2
    
    lines_world = np.zeros_like(lines_px, dtype=np.float64)
    for i in range(len(lines_px)):
        p1_hw = lines_px[i, :2].astype(np.float64)
        p2_hw = lines_px[i, 2:].astype(np.float64)
        
        p1_xy = rot @ p1_hw + t
        p2_xy = rot @ p2_hw + t
        
        # 加回原始偏移 (减去之前的 epsilon + xy_min)
        p1_xy += xy_min - 0.01
        p2_xy += xy_min - 0.01
        
        lines_world[i] = [p1_xy[0], p1_xy[1], p2_xy[0], p2_xy[1]]
    
    # 保存线段结果
    lineseg_path = os.path.join(output_dir, "linesegs.txt")
    with open(lineseg_path, "w") as f:
        for line in lines_world:
            f.write(f"{line[0]:.6f} {line[1]:.6f} {line[2]:.6f} {line[3]:.6f}\n")
    print(f"  已保存 {len(lines_world)} 条线段: {lineseg_path}")
    
    return lines_world


# ============================================================
# 第四部分: 可选 - 直接从 removed.pcd 做简化版（无需 points.txt / info.txt）
# ============================================================

def simple_pcd_to_linesegs(
    pcd_path: str,
    output_dir: str,
    min_height: float = 0.2,
    max_height: float = 1.9,
    scale: int = 60,
    dilate_kernel: int = 3,
):
    """
    简化版: 直接将一个 .pcd 文件转换为 2D 线段。
    不需要 info.txt 和 points.txt，不做语义分离。
    适用于: removed.pcd (已经剔除家具的背景点云) 或任意水平对齐的点云。
    """
    print(f"\n{'='*60}")
    print(f"简化模式: 直接从点云生成 2D 线段")
    print(f"{'='*60}")
    
    print(f"\n读取点云: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"  点数: {len(pcd.points)}, 有颜色: {pcd.has_colors()}")
    
    lines = pcd_to_2d_linesegs(
        pcd, output_dir,
        min_height=min_height,
        max_height=max_height,
        scale=scale,
        dilate_kernel=dilate_kernel,
    )
    
    print(f"\n{'='*60}")
    print(f"完成! 共生成 {len(lines)} 条 2D 线段")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")
    return lines


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="阶段二 → 阶段三: 分割点云 → 2D 线段",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 完整模式: 使用阶段二全部输出 (colorized.pcd + info.txt + points.txt)
  python stage2_to_stage3.py /path/to/stage2_output \\
      --slam /path/to/colorized.pcd \\
      --output pipeline_output

  # 简化模式: 直接用一个 .pcd 文件 (无需语义分离)
  python stage2_to_stage3.py --simple /path/to/removed.pcd \\
      --output pipeline_output
        """,
    )
    
    parser.add_argument(
        "stage2_dir", nargs="?", default=None,
        help="阶段二的输出目录 (包含 info.txt, points.txt, removed.pcd)",
    )
    parser.add_argument(
        "--slam", default=None,
        help="原始 SLAM 点云路径 (colorized.pcd)。如果不提供，将尝试在 stage2_dir 的父目录查找",
    )
    parser.add_argument(
        "--simple", default=None,
        help="简化模式: 直接提供一个 .pcd 文件路径，跳过语义分离",
    )
    parser.add_argument(
        "--output", "-o", default="pipeline_output",
        help="输出目录 (默认: pipeline_output)",
    )
    parser.add_argument("--min-height", type=float, default=0.2, help="墙壁最低高度 (米)")
    parser.add_argument("--max-height", type=float, default=1.9, help="墙壁最高高度 (米)")
    parser.add_argument("--scale", type=int, default=60, help="投影比例 (像素/米)")
    parser.add_argument("--dilate", type=int, default=3, help="膨胀核大小")
    parser.add_argument("--hough-threshold", type=int, default=60, help="Hough 变换阈值")
    parser.add_argument("--hough-min-length", type=int, default=30, help="最小线段长度 (像素)")
    parser.add_argument("--hough-max-gap", type=int, default=30, help="最大线段间隙 (像素)")
    
    args = parser.parse_args()
    
    # ========== 简化模式 ==========
    if args.simple:
        if not os.path.exists(args.simple):
            print(f"[错误] 文件不存在: {args.simple}")
            sys.exit(1)
        simple_pcd_to_linesegs(
            args.simple, args.output,
            min_height=args.min_height,
            max_height=args.max_height,
            scale=args.scale,
            dilate_kernel=args.dilate,
        )
        return
    
    # ========== 完整模式 ==========
    if args.stage2_dir is None:
        parser.print_help()
        print("\n[错误] 请提供阶段二输出目录或使用 --simple 模式")
        sys.exit(1)
    
    stage2_dir = args.stage2_dir
    
    # 检查必需文件
    info_path = os.path.join(stage2_dir, "info.txt")
    points_path = os.path.join(stage2_dir, "points.txt")
    removed_path = os.path.join(stage2_dir, "removed.pcd")
    
    # 查找 SLAM 点云
    slam_path = args.slam
    if slam_path is None:
        # 尝试在父目录查找 colorized.pcd
        parent_dir = os.path.dirname(stage2_dir)
        candidate = os.path.join(parent_dir, "colorized.pcd")
        if os.path.exists(candidate):
            slam_path = candidate
        else:
            # 再上一级
            grandparent = os.path.dirname(parent_dir)
            candidate = os.path.join(grandparent, "colorized.pcd")
            if os.path.exists(candidate):
                slam_path = candidate
    
    print(f"{'='*60}")
    print(f"阶段二 → 阶段三 转换流程")
    print(f"{'='*60}")
    print(f"阶段二输出目录: {stage2_dir}")
    print(f"输出目录: {args.output}")
    
    # 判断使用哪种模式
    has_semantic = os.path.exists(info_path) and os.path.exists(points_path) and slam_path and os.path.exists(slam_path)
    has_removed = os.path.exists(removed_path)
    
    if has_semantic:
        print(f"\n✅ 检测到完整的语义数据，使用语义分离模式")
        print(f"  SLAM 点云: {slam_path}")
        print(f"  info.txt:  {info_path}")
        print(f"  points.txt: {points_path}")
        
        # 语义分离
        walls_pcd, ground_pcd = separate_point_cloud(
            slam_path, info_path, points_path, args.output
        )
        
        # 生成线段
        lines = pcd_to_2d_linesegs(
            walls_pcd, args.output,
            min_height=args.min_height,
            max_height=args.max_height,
            scale=args.scale,
            dilate_kernel=args.dilate,
            hough_threshold=args.hough_threshold,
            hough_min_length=args.hough_min_length,
            hough_max_gap=args.hough_max_gap,
        )
        
    elif has_removed:
        print(f"\n⚠️  未找到完整的语义数据，降级为简化模式 (使用 removed.pcd)")
        print(f"  removed.pcd: {removed_path}")
        
        lines = simple_pcd_to_linesegs(
            removed_path, args.output,
            min_height=args.min_height,
            max_height=args.max_height,
            scale=args.scale,
            dilate_kernel=args.dilate,
        )
    else:
        print(f"\n[错误] 在 {stage2_dir} 中找不到所需文件:")
        print(f"  info.txt:      {'✅' if os.path.exists(info_path) else '❌'}")
        print(f"  points.txt:    {'✅' if os.path.exists(points_path) else '❌'}")
        print(f"  removed.pcd:   {'✅' if has_removed else '❌'}")
        print(f"  colorized.pcd: {'✅' if slam_path and os.path.exists(slam_path) else '❌'}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"✅ 完成! 共生成 {len(lines)} 条 2D 线段")
    print(f"输出文件:")
    for f in sorted(os.listdir(args.output)):
        fpath = os.path.join(args.output, f)
        size = os.path.getsize(fpath)
        print(f"  {f:30s} {size:>10,} bytes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
