# -*- coding: utf-8 -*-
# D:\work\l2bim\preprocess\datasets\pcd_preview_min.py
import os, numpy as np, open3d as o3d
from pathlib import Path
from typing import Optional, Dict

def preview_pcd_interactive(pcd_path: str):
    """本地交互预览（会弹Open3D窗口，适合你自己在电脑上看）"""
    pcd = o3d.io.read_point_cloud(pcd_path)
    if len(pcd.points) == 0:
        raise ValueError(f"[preview] 点云为空或无法读取: {pcd_path}")
    o3d.visualization.draw_geometries([pcd])

def preview_pcd_screenshot(
    pcd_path: str,
    out_path: str,
    width: int = 1600,
    height: int = 1200,
    point_size: float = 2.0,
    bg_white: bool = True,
) -> Dict[str, str]:
    """
    离屏渲染成PNG（适合做API返回或前端展示）
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    if len(pcd.points) == 0:
        raise ValueError(f"[preview] 点云为空或无法读取: {pcd_path}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.array([1,1,1] if bg_white else [0,0,0], dtype=np.float64)
    opt.point_size = point_size
    opt.light_on = True

    vis.poll_events(); vis.update_renderer()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    vis.capture_screen_image(out_path, do_render=True)
    vis.destroy_window()
    return {"png": os.path.abspath(out_path)}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd", required=True)
    ap.add_argument("--mode", choices=["interactive","shot"], default="interactive")
    ap.add_argument("--out", default="./output/preview/preview.png")
    ap.add_argument("--size", type=int, nargs=2, default=[1600,1200])
    ap.add_argument("--pt", type=float, default=2.0)
    ap.add_argument("--bg", choices=["white","black"], default="white")
    args = ap.parse_args()

    if args.mode == "interactive":
        preview_pcd_interactive(args.pcd)
    else:
        preview_pcd_screenshot(
            args.pcd, args.out,
            width=args.size[0], height=args.size[1],
            point_size=args.pt, bg_white=(args.bg=="white")
        )
        print("[saved]", os.path.abspath(args.out))
