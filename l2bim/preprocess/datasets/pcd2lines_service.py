# -*- coding: utf-8 -*-
"""
3D 点云 → 2D 线段检测服务封装
调用现有的：
    - run_pcd2line_once         （单次检测）
    - run_pcd2lines_multistage  （多阶段检测）
输出：
    - 线段结果（.pkl）
    - CAD 可用的 DXF 文件（.dxf）
    - 检测背景图（.png）
"""

import os
import sys
from pathlib import Path

import numpy as np
import cv2
import ezdxf

# === 1. 把项目根目录加入 sys.path，方便 import preprocess.* ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# === 2. 导入你原来项目里的检测函数和 LineSegments ===
from preprocess.datasets.pcd2line_test import run_pcd2line_once
from preprocess.datasets.pcd2lines_step import run_pcd2lines_multistage
from preprocess.geometry.lineseg import LineSegments


def save_linesegs_to_dxf(linesegs: LineSegments, filename: str):
    """
    将 LineSegments 导出为 DXF 矢量线段，方便在 CAD 中使用
    """
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    for line in linesegs.linesegments:
        ax, ay = float(line.point_a[0]), float(line.point_a[1])
        bx, by = float(line.point_b[0]), float(line.point_b[1])
        msp.add_line((ax, ay, 0.0), (bx, by, 0.0))

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc.saveas(filename)


def run_pcd2lines_service(
    pcd_path: str,
    cfg_path: str,
    method: str = "single",
    save_dir: str = "./output/lines",
    save_img: bool = True,
    save_pkl: bool = True,
    save_dxf: bool = True,
):
    """
    封装好的后端接口：点云 → 线段检测 → 导出结果

    Args:
        pcd_path: 输入点云路径 (.pcd / .ply)
        cfg_path: 配置文件 (.yaml)
        method:   "single" 单次检测; "multi" 多阶段检测
        save_dir: 输出目录
        save_img: 是否保存检测用的背景图 (.png)
        save_pkl: 是否保存线段为项目自用格式 (.pkl)
        save_dxf: 是否导出 CAD 可用的 DXF

    Returns:
        dict {
            "method": 使用的检测方法,
            "num_lines": 线段数量,
            "img_path": 背景图路径或 None,
            "pkl_path": pkl 路径或 None,
            "dxf_path": dxf 路径或 None,
        }
    """
    os.makedirs(save_dir, exist_ok=True)

    # === 1. 调用你现有的检测代码 ===
    method = method.lower()
    if method not in ["single", "multi"]:
        raise ValueError(f"method 只能是 'single' 或 'multi'，当前: {method}")

    if method == "single":
        # 对应 line_editor2.py 里的 run_pcd2line_once
        result = run_pcd2line_once(pcd_path, cfg_path, show=False)
        linesegs: LineSegments = result["linesegs"]
        img = result["img"]
    else:
        # 对应 line_editor2.py 里的 run_pcd2lines_multistage
        result = run_pcd2lines_multistage(pcd_path, cfg_path, show=False)
        linesegs: LineSegments = result["lines_all"]
        img = result["img"]

    num_lines = len(linesegs.linesegments)

    img_path = None
    pkl_path = None
    dxf_path = None

    # === 2. 保存背景图（检测用图像） ===
    if save_img and img is not None:
        img_path = os.path.join(save_dir, "lines_debug.png")
        # img 通常是 numpy.uint8，直接保存
        cv2.imwrite(img_path, img)

    # === 3. 保存线段为 pkl（项目内部用） ===
    if save_pkl:
        pkl_path = os.path.join(save_dir, "linesegs.pkl")
        # 这里沿用你项目里的 save_to_file（实际上是 txt，而不是 pickle）
        # 如果你想真·pkl，可以改成 pickle.dump
        linesegs.save_to_file(pkl_path)

    # === 4. 保存线段为 DXF（CAD 用） ===
    if save_dxf:
        dxf_path = os.path.join(save_dir, "linesegs.dxf")
        save_linesegs_to_dxf(linesegs, dxf_path)

    summary = {
        "method": method,
        "num_lines": int(num_lines),
        "img_path": img_path,
        "pkl_path": pkl_path,
        "dxf_path": dxf_path,
    }

    # 打印一份简易 summary，方便命令行调试 / 前端日志查看
    print("\n[pcd2lines_service] 处理完成：")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PCD -> 2D 线段检测服务封装")
    parser.add_argument("--pcd", required=True, help="输入点云路径 (.pcd / .ply)")
    parser.add_argument("--cfg", required=True, help="配置文件路径 (.yaml)")
    parser.add_argument("--method", default="single", choices=["single", "multi"],
                        help="检测方式：single=单次检测; multi=多阶段检测")
    parser.add_argument("--out", default="./output/lines_result", help="输出目录")
    parser.add_argument("--no_img", action="store_true", help="不保存背景图")
    parser.add_argument("--no_pkl", action="store_true", help="不保存 pkl 线段数据")
    parser.add_argument("--no_dxf", action="store_true", help="不保存 DXF CAD 文件")

    args = parser.parse_args()

    run_pcd2lines_service(
        pcd_path=args.pcd,
        cfg_path=args.cfg,
        method=args.method,
        save_dir=args.out,
        save_img=not args.no_img,
        save_pkl=not args.no_pkl,
        save_dxf=not args.no_dxf,
    )
