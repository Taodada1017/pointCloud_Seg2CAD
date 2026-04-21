# -*- coding: utf-8 -*-
"""
3D 点云 → 2D 投影功能封装
只做：RANSAC 去地面 + PCD2d 压平 + PointCloudManager 生成 2D 图
RANSAC 参数直接使用与 pcd2line_test.py 相同的默认值：0.3, 3, 1000
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


# 让 Python 能从项目根目录 import preprocess.*
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.config import load_cfg
from preprocess.geometry.point_cloud import PCD2d, PointCloudManager


def run_pcd_projection(
    pcd_path: str,
    cfg_path: str,
    save_dir: str = "./output/projection",
    distance_threshold: float = 0.3,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    save_png: bool = True,
    projection_scale: float | None = None,
):
    """
    将 3D 点云投影到 2D 平面，并保存 2D 点云与可视化图像。

    Args:
        pcd_path: 输入的 .pcd/.ply 文件路径
        cfg_path: 配置文件 YAML 路径（例如 1f_office_03.yaml）
        save_dir: 输出目录
        distance_threshold, ransac_n, num_iterations: RANSAC 参数（可选）
        save_png: 是否生成 2D 投影 PNG 图像

    Returns:
        dict {
            "walls_2d_pcd": 保存的 2D 点云文件路径,
            "png": 2D 投影图像路径（若 save_png=True）,
            "num_points": 二维点云数量
        }
    """
    t0 = time.perf_counter()

    os.makedirs(save_dir, exist_ok=True)

    # 1. 读取点云
    pcd = o3d.io.read_point_cloud(pcd_path)
    npts = np.asarray(pcd.points).shape[0]
    if npts < 3:
        raise RuntimeError(f"[projection] 点云过少或读取失败: {pcd_path} (points={npts})")

    # 2. 读取 cfg（这个 cfg 之后给 PCD2d / PointCloudManager 用）
    cfg = load_cfg(cfg_path)
    # ⭐ 如果传入新的 scale，则覆盖 YAML 的 pcd2d.scale
    if projection_scale is not None:
        if hasattr(cfg, "pcd2d"):
            cfg.pcd2d.scale = int(projection_scale)
            print(f"[projection] 使用用户自定义投影分辨率 scale = {projection_scale}")
    

    # 3. RANSAC 去地面（参数与 pcd2line_test.py 保持一致）
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    ground_pcd = pcd.select_by_index(inliers)
    walls_pcd = pcd.select_by_index(inliers, invert=True)
    print(f"[projection] after plane seg: ground={len(ground_pcd.points)}, walls={len(walls_pcd.points)}")

    # 某些点云会被误分成“全是地面”，导致墙体为空；投影/检测应该回退继续
    if len(walls_pcd.points) == 0:
        print("[projection] 警告: 去地面后 walls_pcd 为空，回退使用原始点云")
        walls_pcd = pcd
        ground_pcd = o3d.geometry.PointCloud()

    # 4. 使用 PCD2d 做 3D → 2D 压平
    pcd2d = PCD2d(walls_pcd=walls_pcd, ground_pcd=ground_pcd, cfg=cfg)
    ok = pcd2d.get_pcd2d()
    if not ok or pcd2d.walls_2d is None or len(pcd2d.walls_2d.points) == 0:
        raise RuntimeError(
            "[projection] PCD2d.get_pcd2d() 失败或二维点云为空。"
            "可能原因：cfg 的 pcd2d.min_height/max_height 把点裁掉，或者点云本身没有足够墙体点。"
        )

    walls_2d = pcd2d.walls_2d
    pts0 = np.asarray(walls_2d.points)
    if pts0.shape[0] > 0:
        zmin, zmax = float(pts0[:, 2].min()), float(pts0[:, 2].max())
        print(f"[projection] walls_2d points={pts0.shape[0]}, z_range=[{zmin:.6f},{zmax:.6f}]")

    # ⭐ 4.1 关键一步：把二维点云平移到正坐标（仿照 pcd2line_test.py）
    pts = np.asarray(walls_2d.points)
    T_min2og = np.eye(4)
    xy_min = pts.min(axis=0)
    T_min2og[0, 3] = -xy_min[0]
    T_min2og[1, 3] = -xy_min[1]
    walls_2d.transform(T_min2og)

    

    # 5. 保存 2D 点云（已经平移后的）
    out_pcd_path = os.path.join(save_dir, "walls_2d.pcd")
    o3d.io.write_point_cloud(out_pcd_path, walls_2d)

    # 6. 使用 PointCloudManager 生成 PNG 投影
    walls_np = np.asarray(walls_2d.points)
    pc_manager = PointCloudManager(walls_np, cfg)
    pc_manager.get_img()

    # ⭐ 6.1 仿照原脚本：再反一次色，让点变白、背景黑
    img_np = np.array(pc_manager.png)
    img_np = 255 - img_np   # 和 pcd2line_test.py 里一样，再反色一次

    img_pil = Image.fromarray(img_np)

    out_png_path = os.path.join(save_dir, "projection.png")
    if save_png:
        img_pil.save(out_png_path)

    total_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[timing] total: {total_ms:.2f} ms")


    print(f"[projection] 2D 点云数量: {walls_np.shape[0]}")
    print(f"[projection] 保存 2D 点云: {out_pcd_path}")
    if save_png:
        print(f"[projection] 保存投影图: {out_png_path}")

    return {
        "walls_2d_pcd": out_pcd_path,
        "png": out_png_path if save_png else None,
        "num_points": int(walls_np.shape[0]),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PCD -> 2D 投影")
    parser.add_argument("--pcd", required=True, help="输入点云路径 (.pcd/.ply)")
    parser.add_argument("--cfg", required=True, help="配置文件路径 (.yaml)")
    parser.add_argument("--out", default="./output/projection", help="输出目录")
    parser.add_argument("--no_png", action="store_true", help="不保存 PNG 图像")
    parser.add_argument("--scale", type=float, help="投影分辨率（覆盖 YAML 中的 pcd2d.scale）")
    args = parser.parse_args()

    result = run_pcd_projection(
        pcd_path=args.pcd,
        cfg_path=args.cfg,
        save_dir=args.out,
        save_png=not args.no_png,
        projection_scale=args.scale,
    )

    print("\n===== 投影完成 =====")
    print(result)
