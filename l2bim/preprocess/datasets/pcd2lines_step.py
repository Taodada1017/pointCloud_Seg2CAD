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

# ---- 配置读取 ----
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

from preprocess.utils import setup_plot
from preprocess.geometry.point_cloud import PCD2d, PointCloudManager

# hough 检测
try:
    from preprocess.datasets.lineseg_manager import hough_line_detection
except Exception:
    from lineseg_manager import hough_line_detection

from preprocess.geometry.lineseg import LineSegments

# ---------- 可复用函数 ----------
def run_pcd2lines_multistage(
    pcd_path: str,
    cfg_path: str,
    save_dir: str = None,
    distance_threshold: float = 0.3,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    show: bool = False,
):
    """
    多阶段检测流程：长线(严格) -> 掩膜 -> 短线(宽松) -> 合并
    返回 dict：{pcd, ground_pcd, walls_pcd, walls_2d, img, lines_long, lines_short, lines_all}
    """
    t0 = time.time()
    from easydict import EasyDict
    cfg = _safe_load_cfg(cfg_path)
    cfg = EasyDict(cfg)

    pcd = o3d.io.read_point_cloud(pcd_path)

    # 1) RANSAC 去地面
    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    ground_pcd = pcd.select_by_index(inliers)
    walls_pcd = pcd.select_by_index(inliers, invert=True)

    # 2) 3D -> 2D
    pcd2d = PCD2d(walls_pcd=walls_pcd, ground_pcd=ground_pcd, cfg=cfg)
    pcd2d.get_pcd2d()
    walls_2d = pcd2d.walls_2d

    T_min2og = np.eye(4)
    xy_min = np.min(np.asarray(walls_2d.points), axis=0)
    T_min2og[0, 3] = -xy_min[0]
    T_min2og[1, 3] = -xy_min[1]
    walls_2d.transform(T_min2og)

    pcm = PointCloudManager(np.asarray(walls_2d.points), cfg)
    pcm.get_img()
    img = 255 - np.array(pcm.png)

    # 3) Stage-1: 严格参数找长线
    lines_long = hough_line_detection(img, threshold=120, minLineLength=100, maxLineGap=10)
    lines_long.merge_uf(5, 5)
    lines_long.cluster_angle(bandwidth=0.01)
    lines_long.remove_minor_angle(min_num=5, min_length=100)

    # 4) 掩膜后再检测短线
    img_masked = _mask_lines(img, lines_long, color=0, thickness=15)
    lines_short = hough_line_detection(img_masked, threshold=60, minLineLength=30, maxLineGap=10)
    lines_short.merge_uf(15, 15)
    lines_short.cluster_angle(bandwidth=0.05)
    lines_short.remove_minor_angle(min_num=5, min_length=40)

    # 5) 合并
    lines_all = LineSegments(lines_long.linesegments + lines_short.linesegments)
    lines_all.merge_uf(10, 10)
    lines_all.cluster_angle(bandwidth=0.01)
    lines_all.remove_minor_angle(min_num=5, min_length=30)

    # 6) 可视化与保存
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig1, ax1 = _draw_lines(img, lines_long); (Path(save_dir)/"stage1_long.png").write_bytes(_save_fig(fig1)); plt.close(fig1)
        fig2, ax2 = _draw_lines(img_masked, lines_short); (Path(save_dir)/"stage2_short.png").write_bytes(_save_fig(fig2)); plt.close(fig2)
        fig3, ax3 = _draw_lines(img, lines_all); (Path(save_dir)/"final_merge.png").write_bytes(_save_fig(fig3)); plt.close(fig3)

    if show:
        for im, ls in [(img, lines_long), (img_masked, lines_short), (img, lines_all)]:
            fig, ax = _draw_lines(im, ls); plt.show(); plt.close(fig)

    print(f"[multi] long:{len(lines_long.linesegments)}, short:{len(lines_short.linesegments)}, "
          f"final:{len(lines_all.linesegments)}, time:{time.time()-t0:.2f}s")

    return dict(
        pcd=pcd, ground_pcd=ground_pcd, walls_pcd=walls_pcd,
        walls_2d=walls_2d, img=img,
        lines_long=lines_long, lines_short=lines_short, lines_all=lines_all
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

def _mask_lines(img, linesegs, color=0, thickness=15):
    im = img.copy()
    for L in linesegs.linesegments:
        x1, y1 = map(int, L.point_a); x2, y2 = map(int, L.point_b)
        cv2.line(im, (x1, y1), (x2, y2), color, thickness)
    return im

def _save_fig(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight", pad_inches=0)
    return buf.getvalue()

# ---------- 命令行入口 ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", default="D:/work/l2bim/output")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    run_pcd2lines_multistage(args.pcd, args.cfg, save_dir=args.out, show=args.show)
