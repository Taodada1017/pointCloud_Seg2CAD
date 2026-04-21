"""Batch convert NPY point clouds to PCD (ASCII).

Default behavior:
- Scan all .npy files under D:\\work\\l2bim\\npy recursively
- Convert each to .pcd
- Save to D:\\work\\l2bim\\output_pcd with the same relative folder structure
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_INPUT_ROOT = Path(r"D:\work\l2bim\npy")
DEFAULT_OUTPUT_ROOT = Path(r"D:\work\l2bim\output_pcd")


def _extract_xyz(arr: np.ndarray, src_path: Path) -> np.ndarray:
	"""Extract xyz as float32 from common NPY point-cloud layouts."""
	arr = np.asarray(arr)

	if arr.ndim != 2:
		raise ValueError(f"Expected 2D array for point cloud, got shape={arr.shape} in {src_path}")

	# Standard layout: (N, C), C >= 3
	if arr.shape[1] >= 3:
		xyz = arr[:, :3]
	# Alternate layout: (3, N) or (C, N), C >= 3
	elif arr.shape[0] >= 3:
		xyz = arr[:3, :].T
	else:
		raise ValueError(f"Array must have at least 3 coordinate channels, got shape={arr.shape} in {src_path}")

	xyz = xyz.astype(np.float32, copy=False)
	if xyz.ndim != 2 or xyz.shape[1] != 3:
		raise ValueError(f"Failed to normalize xyz to (N, 3), got {xyz.shape} in {src_path}")

	return xyz


def _iter_npy_files(input_root: Path) -> Iterable[Path]:
	# Use suffix check so both .npy and .NPY are handled.
	return sorted(
		p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() == ".npy"
	)


def write_pcd_ascii(output_pcd: Path, xyz: np.ndarray) -> None:
	"""Write xyz points to ASCII PCD with fields x y z."""
	n_points = int(xyz.shape[0])
	output_pcd.parent.mkdir(parents=True, exist_ok=True)

	with output_pcd.open("w", encoding="utf-8") as f:
		f.write("# .PCD v0.7 - Point Cloud Data file format\n")
		f.write("VERSION 0.7\n")
		f.write("FIELDS x y z\n")
		f.write("SIZE 4 4 4\n")
		f.write("TYPE F F F\n")
		f.write("COUNT 1 1 1\n")
		f.write(f"WIDTH {n_points}\n")
		f.write("HEIGHT 1\n")
		f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
		f.write(f"POINTS {n_points}\n")
		f.write("DATA ascii\n")

		for x, y, z in xyz:
			f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def convert_npy_to_pcd(input_npy: Path, output_pcd: Path) -> None:
	arr = np.load(input_npy, allow_pickle=False)
	xyz = _extract_xyz(arr, input_npy)
	write_pcd_ascii(output_pcd, xyz)


def convert_all_npy_to_pcd(input_root: Path, output_root: Path) -> None:
	if not input_root.exists():
		raise FileNotFoundError(f"Input root not found: {input_root}")

	output_root.mkdir(parents=True, exist_ok=True)

	npy_files = list(_iter_npy_files(input_root))
	if not npy_files:
		print(f"No .npy files found under: {input_root}")
		return

	folders = {src.parent for src in npy_files}
	print(f"Found {len(folders)} folders containing npy files under: {input_root}")

	print(f"Found {len(npy_files)} npy files under: {input_root}")

	success = 0
	fail = 0
	for src in npy_files:
		rel_path = src.relative_to(input_root)
		dst = (output_root / rel_path).with_suffix(".pcd")
		try:
			convert_npy_to_pcd(src, dst)
			success += 1
			print(f"[OK] {src} -> {dst}")
		except Exception as exc:
			fail += 1
			print(f"[FAILED] {src}: {exc}")

	print(f"Done. success={success}, failed={fail}, output_root={output_root}")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Batch convert NPY point clouds to PCD")
	parser.add_argument(
		"--input-root",
		type=Path,
		default=DEFAULT_INPUT_ROOT,
		help=f"Input root folder containing .npy files (default: {DEFAULT_INPUT_ROOT})",
	)
	parser.add_argument(
		"--output-root",
		type=Path,
		default=DEFAULT_OUTPUT_ROOT,
		help=f"Output root folder for .pcd files (default: {DEFAULT_OUTPUT_ROOT})",
	)
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()
	convert_all_npy_to_pcd(args.input_root, args.output_root)


if __name__ == "__main__":
	main()
