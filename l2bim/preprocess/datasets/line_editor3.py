# -*- coding: utf-8 -*-
"""
line_editor3.py
线段编辑“核心算法模块”，不包含任何 GUI 代码，供前端自己的界面调用。

功能：
1. 从点云 + 配置文件运行线段检测（单次 / 多阶段）
2. 保存和维护 LineSegments（线段集合）
3. 提供：获取线段列表、移动端点、删除线段、保存结果(pkl/dxf/json) 等接口
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
import ezdxf

# ------------------------------------------------------------------
# 把项目根目录加入 sys.path，方便 import preprocess.*
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.datasets.pcd2line_test import run_pcd2line_once
from preprocess.datasets.pcd2lines_step import run_pcd2lines_multistage
from preprocess.geometry.lineseg import LineSegments


class LineEditModel:
    """
    线段编辑的“数据 + 算法”核心类（无 GUI）

    前端可以这样用：

        model = LineEditModel()
        model.load_from_pcd_cfg(pcd_path, cfg_path, method="single")

        img  = model.get_image()   # numpy 图像，用来画背景
        lines = model.get_lines()  # 线段列表，用来画线 + 端点

        # 拖动端点
        model.move_point(line_idx=0, point_idx=0, new_x=123.4, new_y=56.7)

        # 删除一条线
        model.delete_line(3)

        # 保存
        model.save_dxf("xxx.dxf")
        model.save_json("xxx.json")
    """

    def __init__(self):
        self.img: Optional[np.ndarray] = None
        self.linesegs: Optional[LineSegments] = None
        self.method: str = "single"

    # ----------------- 1. 从 pcd + cfg 运行检测 -----------------
    def load_from_pcd_cfg(self, pcd_path: str, cfg_path: str, method: str = "single"):
        """
        从点云和配置文件运行线段检测，内部调用项目已有的函数。

        Args:
            pcd_path: .pcd / .ply 路径
            cfg_path: .yaml 路径
            method:   "single" 或 "multi"
        """
        method = method.lower()
        if method not in ["single", "multi"]:
            raise ValueError("method 必须是 'single' 或 'multi'")

        self.method = method

        if method == "single":
            result = run_pcd2line_once(pcd_path, cfg_path, show=False)
            self.linesegs = result["linesegs"]
            self.img = result["img"]
        else:
            result = run_pcd2lines_multistage(pcd_path, cfg_path, show=False)
            self.linesegs = result["lines_all"]
            self.img = result["img"]

        if self.linesegs is None:
            raise RuntimeError("线段检测失败：linesegs 为空")

    # ----------------- 2. 获取结果 -----------------
    def get_image(self) -> np.ndarray:
        """返回背景图像（numpy 数组），前端用来画底图。"""
        if self.img is None:
            raise RuntimeError("尚未加载图像，请先调用 load_from_pcd_cfg")
        return self.img

    def get_lines(self) -> List[Dict]:
        """
        返回当前所有线段的列表（纯 Python dict，便于前端使用）

        每个元素：
            {
              "id": int,
              "ax": float, "ay": float,
              "bx": float, "by": float,
              "length": float
            }
        坐标单位：与检测时的图像坐标一致（和 line_editor2.py 相同）
        """
        if self.linesegs is None:
            raise RuntimeError("尚未加载线段，请先调用 load_from_pcd_cfg")

        data = []
        for idx, line in enumerate(self.linesegs.linesegments):
            ax, ay = float(line.point_a[0]), float(line.point_a[1])
            bx, by = float(line.point_b[0]), float(line.point_b[1])
            item = {
                "id": idx,
                "ax": ax,
                "ay": ay,
                "bx": bx,
                "by": by,
                "length": float(line.get_length()),
            }
            data.append(item)
        return data

    # ----------------- 3. 编辑操作：移动端点 / 覆盖线段 / 删除 -----------------
    def _check_index(self, line_idx: int):
        if self.linesegs is None:
            raise RuntimeError("尚未加载线段，请先调用 load_from_pcd_cfg")
        if not (0 <= line_idx < len(self.linesegs.linesegments)):
            raise IndexError(f"line_idx 越界：{line_idx}")

    def move_point(self, line_idx: int, point_idx: int, new_x: float, new_y: float):
        """
        移动某条线段的端点（对应你 GUI 里的拖动端点）

        Args:
            line_idx: 线段索引
            point_idx: 0 = 端点 A, 1 = 端点 B
            new_x, new_y: 新坐标（与 get_lines 返回的坐标同一坐标系）
        """
        self._check_index(line_idx)
        if point_idx not in (0, 1):
            raise ValueError("point_idx 只能是 0 (A) 或 1 (B)")

        line = self.linesegs.linesegments[line_idx]
        if point_idx == 0:
            line.point_a = np.array([new_x, new_y], dtype=float)
        else:
            line.point_b = np.array([new_x, new_y], dtype=float)

        # 更新方向和长度（和 line_editor2.py 里完全一致）
        line.direction = line.point_b - line.point_a
        line.direction = line.direction.astype(float)
        if np.linalg.norm(line.direction) > 0:
            line.direction /= np.linalg.norm(line.direction)
        line.length = line.get_length()

    def update_line(self, line_idx: int, ax: float, ay: float, bx: float, by: float):
        """
        用新的端点坐标直接覆盖一条线（对应“在属性面板输入坐标”的情况）
        """
        self._check_index(line_idx)
        line = self.linesegs.linesegments[line_idx]
        line.point_a = np.array([ax, ay], dtype=float)
        line.point_b = np.array([bx, by], dtype=float)
        line.direction = line.point_b - line.point_a
        line.direction = line.direction.astype(float)
        if np.linalg.norm(line.direction) > 0:
            line.direction /= np.linalg.norm(line.direction)
        line.length = line.get_length()

    def delete_line(self, line_idx: int):
        """删除指定索引的线段"""
        self._check_index(line_idx)
        del self.linesegs.linesegments[line_idx]

    # ----------------- 4. 保存接口：pkl / dxf / json -----------------
    def save_pkl(self, filename: str):
        """保存为项目原有的 pkl/txt 格式（直接复用 LineSegments.save_to_file）"""
        if self.linesegs is None:
            raise RuntimeError("没有线段可保存")
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        self.linesegs.save_to_file(filename)

    def save_dxf(self, filename: str):
        """保存为 CAD 可用的 DXF 线段文件"""
        if self.linesegs is None:
            raise RuntimeError("没有线段可保存")
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        
        # 获取图像高度以进行坐标翻转（如果有图像的话）
        img_height = 0
        if hasattr(self, 'img') and self.img is not None:
            img_height = self.img.shape[0]
        
        for line in self.linesegs.linesegments:
            ax, ay = float(line.point_a[0]), float(line.point_a[1])
            bx, by = float(line.point_b[0]), float(line.point_b[1])
            
            # 翻转y坐标以匹配PNG图片的显示方向（从屏幕坐标系转换为笛卡尔坐标系时需要翻转）
            # 注意：这里只在有图像高度信息时才进行翻转
            if img_height > 0:
                ay = img_height - ay
                by = img_height - by
            
            msp.add_line((ax, ay, 0.0), (bx, by, 0.0))
        doc.saveas(filename)

    def save_json(self, filename: str):
        """保存为 JSON（给前端 / 其他语言使用）"""
        data = self.get_lines()
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------
# 作为脚本运行时：简单测试 + 命令行接口（方便你本地验证）
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="线段编辑核心模块测试：点云 -> 线段 -> 输出 dxf/json"
    )
    parser.add_argument("--pcd", required=True, help="输入点云路径 (.pcd/.ply)")
    parser.add_argument("--cfg", required=True, help="配置文件路径 (.yaml)")
    parser.add_argument(
        "--method",
        default="single",
        choices=["single", "multi"],
        help="检测方式：single=单次检测; multi=多阶段检测",
    )
    parser.add_argument(
        "--out", default="./output/line_edit_core_test", help="输出目录"
    )

    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    model = LineEditModel()
    model.load_from_pcd_cfg(args.pcd, args.cfg, method=args.method)

    # 保存图像、dxf、json 作为测试结果
    img = model.get_image()
    img_path = os.path.join(out_dir, "bg.png")
    cv2.imwrite(img_path, img)

    dxf_path = os.path.join(out_dir, "lines.dxf")
    json_path = os.path.join(out_dir, "lines.json")
    pkl_path = os.path.join(out_dir, "lines.pkl")

    model.save_dxf(dxf_path)
    model.save_json(json_path)
    model.save_pkl(pkl_path)

    print(f"[line_edit_core] 完成：共 {len(model.get_lines())} 条线段")
    print(f"  背景图: {img_path}")
    print(f"  DXF:     {dxf_path}")
    print(f"  JSON:    {json_path}")
    print(f"  PKL:     {pkl_path}")
