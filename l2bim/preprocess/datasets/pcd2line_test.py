# -*- coding: utf-8 -*-
import sys, time, cv2, numpy as np, open3d as o3d, matplotlib.pyplot as plt
from pathlib import Path

import sys
import os

current_file = os.path.abspath(__file__)
# 向上跳三级：datasets → preprocess → 根目录
root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
#print("修正后root_path:", root_path)  # 验证是否为项目根目录
sys.path.append(root_path)

# ---- 配置读取：优先使用 preprocess/cfg_loader.py 的 load_cfg；没有则内置读取 ----
def _safe_load_cfg(cfg_path):
    try:
        from preprocess.cfg_loader import load_cfg as _loader
        return _loader(cfg_path)
    except Exception:
        import yaml, os
        if not Path(cfg_path).exists():
            raise FileNotFoundError(f"[config] 配置文件不存在: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print(f"[config] 已加载配置文件: {cfg_path}")
        return cfg

# ---- 其余模块 ----
from preprocess.utils import setup_plot
from preprocess.geometry.point_cloud import PCD2d, PointCloudManager

# hough 检测：优先从 preprocess.geometry.lineseg_manager 导入，失败再退回同级
try:
    from preprocess.datasets.lineseg_manager import hough_line_detection
except Exception:
    from lineseg_manager import hough_line_detection  # 若你的文件就在同目录

# ---------- 可复用函数 ----------
def run_pcd2line_once(
    pcd_path: str,
    cfg_path: str,
    save_dir: str = None,
    distance_threshold: float = 0.3,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    show: bool = False,
    min_line_length: int = 30, 
):
    """
    一次检测流程：PCD -> 去地面 -> 2D投影 -> 霍夫线段 -> 合并/筛选 -> 叠画并保存
    返回 dict：{pcd, ground_pcd, walls_pcd, walls_2d, img, linesegs, corners}
    """
    t0 = time.time()
    from easydict import EasyDict
    cfg = _safe_load_cfg(cfg_path)
    cfg = EasyDict(cfg)

    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd is None or len(pcd.points) == 0:
        raise RuntimeError(f"点云为空或无法读取: {pcd_path}")

    # 1) RANSAC 去地面
    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    ground_pcd = pcd.select_by_index(inliers)
    walls_pcd = pcd.select_by_index(inliers, invert=True)
    print(f"[pcd2line_test] after plane seg: ground={len(ground_pcd.points)}, walls={len(walls_pcd.points)}")

    # 有些点云/参数组合会把所有点都分到地面，导致墙体为空；这里做兜底
    if len(walls_pcd.points) == 0:
        print("[pcd2line_test] 警告: 去地面后 walls_pcd 为空，回退使用原始点云进行后续处理")
        walls_pcd = pcd
        ground_pcd = o3d.geometry.PointCloud()

    # 2) 3D -> 2D
    pcd2d = PCD2d(walls_pcd=walls_pcd, ground_pcd=ground_pcd, cfg=cfg)
    pcd2d.get_pcd2d()
    walls_2d = pcd2d.walls_2d

    pts2d = np.asarray(walls_2d.points) if walls_2d is not None else np.empty((0, 3))
    if pts2d.shape[0] > 0:
        zmin, zmax = float(pts2d[:, 2].min()), float(pts2d[:, 2].max())
        print(f"[pcd2line_test] walls_2d points={pts2d.shape[0]}, z_range=[{zmin:.6f},{zmax:.6f}]")

    if walls_2d is None or len(walls_2d.points) == 0:
        raise RuntimeError(
            "2D 投影后墙体点为空，无法进行线段检测。"
            "\n可能原因：点云确实没有墙体/全是地面；或者 cfg 的投影/过滤参数不适配当前点云。"
            "\n建议：调大 distance_threshold 或减小 ransac 迭代；或先用 /preview/v2 检查点云是否正常。"
        )

    # 平移到正坐标
    T_min2og = np.eye(4)
    xy_min = np.min(np.asarray(walls_2d.points), axis=0)
    T_min2og[0, 3] = -xy_min[0]
    T_min2og[1, 3] = -xy_min[1]
    walls_2d.transform(T_min2og)

    # 3) 生成 2D 栅格图像
    pcm = PointCloudManager(np.asarray(walls_2d.points), cfg)
    pcm.get_img()
    img = 255 - np.array(pcm.png)
    print(f"[pcd2line_test] img.shape={img.shape}, dtype={img.dtype}")

    # 4) 线段检测（一次）
    linesegs = hough_line_detection(
        img,
        threshold=60,
        minLineLength=min_line_length,  # ⭐ 使用用户传入的长度
        maxLineGap=30,
    )

    linesegs.merge_uf(angle_thd=10, pt_thd=10)
    linesegs.cluster_angle(bandwidth=0.01)
    linesegs.remove_minor_angle(min_num=5, min_length=60)

    # 5) 角点（可选）
    try:
        corners = linesegs.intersections(threshold=0.1)
    except Exception:
        corners = []

    # 6) 可视化与保存
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = _draw_lines(img, linesegs)
        (Path(save_dir) / "once_detect.png").write_bytes(_save_fig(fig))
        plt.close(fig)

    if show:
        fig, ax = _draw_lines(img, linesegs)
        plt.show()
        plt.close(fig)

    print(f"[once] lines: {len(linesegs.linesegments)}, time: {time.time()-t0:.2f}s")
    return dict(
        pcd=pcd, ground_pcd=ground_pcd, walls_pcd=walls_pcd,
        walls_2d=walls_2d, img=img, linesegs=linesegs, corners=corners
    )

# ---------- 小工具 ----------
def _draw_lines(img, linesegs):
    fig, ax = setup_plot(img.shape)
    ax.imshow(img, cmap="gray")
    for L in linesegs.linesegments:
        x1, y1 = L.point_a
        x2, y2 = L.point_b
        ax.plot([x1, x2], [y1, y2], "r", linewidth=2)
    return fig, ax

def _save_fig(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight", pad_inches=0)
    return buf.getvalue()

# ---------- 命令行入口 ----------
if __name__ == "__main__":
    config_path=r"D:\work\l2bim\configs\interval\15m\1F\1f_office_03.yaml"
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", default="D:/work/l2bim/output")
    ap.add_argument("--show", action="store_true")
    ap.add_argument(
        "--min_len",
        type=int,
        default=30,
        help="最小线段长度（像素，越大保留的线段越长）",
    )
    args = ap.parse_args()

    run_pcd2line_once(
        args.pcd,
        args.cfg,
        save_dir=args.out,
        show=args.show,
        min_line_length=args.min_len,  # ⭐ 传给检测函数
    )
