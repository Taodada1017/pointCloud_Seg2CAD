"""Convert point cloud data from TXT to PCD (ASCII).

Supported TXT layouts:
1) Columns: x y z
2) Columns: x y z r g b (or any layout with >= 6 columns, RGB in cols 4-6)
3) Extra columns are ignored unless --use-rgb is enabled
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _to_float_xyz(points: np.ndarray) -> np.ndarray:
    """Convert xyz to float32 with strict shape check."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"xyz shape must be (N, 3), got {points.shape}")
    return points.astype(np.float32, copy=False)


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    """Normalize RGB to uint8. Supports 0-1 or 0-255 values."""
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb shape must be (N, 3), got {rgb.shape}")

    rgb = rgb.astype(np.float32, copy=False)
    max_val = float(np.nanmax(rgb)) if rgb.size else 0.0
    if max_val <= 1.0:
        rgb = rgb * 255.0
    rgb = np.clip(rgb, 0, 255)
    return rgb.astype(np.uint8)


def _fast_loadtxt(txt_path: Path, usecols: Tuple[int, ...]) -> np.ndarray:
    """Fast numeric loading for large TXT files with whitespace/comma fallback."""
    try:
        data = np.loadtxt(
            txt_path,
            dtype=np.float64,
            comments="#",
            usecols=usecols,
        )
    except Exception:
        data = np.loadtxt(
            txt_path,
            dtype=np.float64,
            comments="#",
            delimiter=",",
            usecols=usecols,
        )

    if data.size == 0:
        raise ValueError(f"No numeric data found in file: {txt_path}")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data


def parse_txt_point_cloud(txt_path: Path, use_rgb: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    xyz_data = _fast_loadtxt(txt_path, usecols=(0, 1, 2))
    xyz = _to_float_xyz(xyz_data)

    rgb = None
    if use_rgb:
        rgb_data = _fast_loadtxt(txt_path, usecols=(3, 4, 5))
        if len(rgb_data) != len(xyz):
            raise ValueError("RGB columns do not match XYZ row count")
        rgb = _to_uint8_rgb(rgb_data)

    return xyz, rgb


def write_pcd_ascii(pcd_path: Path, xyz: np.ndarray, rgb: Optional[np.ndarray] = None) -> None:
    if rgb is not None and len(rgb) != len(xyz):
        raise ValueError("xyz and rgb must have the same number of points")

    n_points = xyz.shape[0]

    with pcd_path.open("w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        if rgb is None:
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
        else:
            f.write("FIELDS x y z r g b\n")
            f.write("SIZE 4 4 4 1 1 1\n")
            f.write("TYPE F F F U U U\n")
            f.write("COUNT 1 1 1 1 1 1\n")

        f.write(f"WIDTH {n_points}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n_points}\n")
        f.write("DATA ascii\n")

        if rgb is None:
            for x, y, z in xyz:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        else:
            for (x, y, z), (r, g, b) in zip(xyz, rgb):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def convert_txt_to_pcd(input_txt: Path, output_pcd: Path, use_rgb: bool = False) -> None:
    xyz, rgb = parse_txt_point_cloud(input_txt, use_rgb=use_rgb)
    write_pcd_ascii(output_pcd, xyz, rgb)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert point cloud TXT file to PCD")
    parser.add_argument("input", type=Path, help="Input .txt path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .pcd path (default: same stem as input)",
    )
    parser.add_argument(
        "--use-rgb",
        action="store_true",
        help="Read columns 4-6 as RGB (requires at least 6 columns)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_txt: Path = args.input
    output_pcd: Path = args.output if args.output is not None else input_txt.with_suffix(".pcd")

    if not input_txt.exists():
        raise FileNotFoundError(f"Input file not found: {input_txt}")

    convert_txt_to_pcd(input_txt, output_pcd, use_rgb=args.use_rgb)
    print(f"Converted: {input_txt} -> {output_pcd}")


if __name__ == "__main__":
    main()
