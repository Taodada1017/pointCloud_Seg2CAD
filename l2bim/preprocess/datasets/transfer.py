# ...existing code...
#!/usr/bin/env python3
"""
把 .las/.laz 批量转换为 .pcd，默认输出到当前工作目录下的 pcds 文件夹。
依赖: laspy, open3d, numpy, tqdm
"""
import os
import argparse
import sys
from pathlib import Path

def has_modules():
    try:
        import laspy, open3d as o3d, numpy as np, tqdm
        return True
    except Exception as e:
        print("缺少依赖：", e)
        print("当前 Python：", sys.executable)
        return False

def convert_file(las_path: Path, out_dir: Path, overwrite: bool = False, verbose: bool = False):
    import laspy
    import numpy as np
    import open3d as o3d
    import shutil

    out_path = out_dir / (las_path.stem + ".pcd")
    if out_path.exists() and not overwrite:
        return str(out_path), False

    if verbose:
        print(f"输入：{las_path}")
        print(f"输出：{out_path}")

    def _sniff_magic(path: Path) -> bytes:
        try:
            with open(path, "rb") as f:
                return f.read(64)
        except Exception:
            return b""

    def _fallback_open3d_read_write(reason: str, magic: bytes | None = None):
        if verbose:
            head = magic if magic is not None else _sniff_magic(las_path)
            head_preview = head[:32].replace(b"\r", b"\\r").replace(b"\n", b"\\n")
            print(f"laspy 读取失败，尝试按通用点云格式读取（原因：{reason}）")
            print(f"文件头前 32 bytes：{head_preview!r}")

        head = magic if magic is not None else _sniff_magic(las_path)
        # 如果内容就是 PCD（但扩展名可能被改成 .las），直接复制成 .pcd 即可
        if head.startswith(b"# .PCD") or b"\nVERSION" in head or head.startswith(b"VERSION"):
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(las_path, out_path)
            if verbose:
                size = out_path.stat().st_size
                print(f"检测到 PCD 内容，直接复制生成：{out_path}（{size} bytes）")
            return str(out_path), True

        pcd_any = o3d.io.read_point_cloud(str(las_path))
        if pcd_any is None or len(pcd_any.points) == 0:
            raise RuntimeError(
                f"laspy 读取失败且 open3d 也无法读取该文件：{las_path}。"
                "若这是 PCD/PLY/XYZ 等格式，请确认扩展名与内容一致。"
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        ok_any = o3d.io.write_point_cloud(str(out_path), pcd_any, write_ascii=False)
        if not ok_any:
            ok_any = o3d.io.write_point_cloud(str(out_path), pcd_any, write_ascii=True)
        if not ok_any:
            raise RuntimeError(f"写入失败：{out_path}（open3d 回退写入返回 False）")
        if verbose:
            print(f"回退读取/写入成功：{out_path}（points={len(pcd_any.points)}）")
        return str(out_path), True

    try:
        las = laspy.read(str(las_path))
    except Exception as e:
        magic = _sniff_magic(las_path)
        return _fallback_open3d_read_write(str(e), magic=magic)

    pts = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    if pts.size == 0:
        raise RuntimeError(f"点为空：{las_path}")

    finite_mask = np.isfinite(pts).all(axis=1)
    if not bool(finite_mask.all()):
        before = int(pts.shape[0])
        pts = pts[finite_mask]
        after = int(pts.shape[0])
        if verbose:
            print(f"过滤无效点(NaN/Inf)：{before} -> {after}")
        if pts.size == 0:
            raise RuntimeError(f"过滤 NaN/Inf 后点为空：{las_path}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # 尝试读取 RGB
    try:
        if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
            r = np.asarray(las.red, dtype=np.float64)
            g = np.asarray(las.green, dtype=np.float64)
            b = np.asarray(las.blue, dtype=np.float64)
            if r.shape[0] == finite_mask.shape[0]:
                r = r[finite_mask]
                g = g[finite_mask]
                b = b[finite_mask]
            # 归一化到 0-1（las 常为 0-65535 或 0-255）
            rgb_max = max(r.max() if r.size else 1, g.max() if g.size else 1, b.max() if b.size else 1)
            colors = np.vstack((r, g, b)).T / float(rgb_max)
            pcd.colors = o3d.utility.Vector3dVector(colors)
    except Exception:
        pass

    out_dir.mkdir(parents=True, exist_ok=True)

    ok = o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False)
    if not ok:
        if verbose:
            print("二进制 PCD 写入失败，尝试 ASCII...")
        ok = o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=True)
    if not ok:
        raise RuntimeError(f"写入失败：{out_path}（open3d.io.write_point_cloud 返回 False）")

    if verbose:
        try:
            size = out_path.stat().st_size
            print(f"写入成功：{out_path}（{size} bytes, points={len(pcd.points)}）")
        except Exception:
            print(f"写入成功：{out_path}（points={len(pcd.points)}）")
    return str(out_path), True

def find_las_files(root: Path, recursive: bool = True):
    exts = ["*.las", "*.laz"]
    if recursive:
        for ext in exts:
            yield from root.rglob(ext)
    else:
        for ext in exts:
            yield from root.glob(ext)

def main():
    parser = argparse.ArgumentParser(description="批量把 .las/.laz 转为 .pcd")
    parser.add_argument("-i", "--input", type=str, default=".", help="输入目录（默认当前目录）")
    parser.add_argument("-o", "--output", type=str, default=None, help="输出目录（默认 ./pcds）")
    parser.add_argument("-r", "--recursive", action="store_true", help="递归查找")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 .pcd")
    parser.add_argument("--verbose", action="store_true", help="打印详细日志")
    args = parser.parse_args()

    if not has_modules():
        print("请先安装依赖：pip install laspy open3d numpy tqdm")
        sys.exit(1)

    from tqdm import tqdm

   # ...existing code...

    in_path = Path(args.input).resolve()
    out_dir = Path(args.output).resolve() if args.output else (Path.cwd() / "pcds")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"输入不存在：{in_path}")
        sys.exit(2)

    if in_path.is_file():
        files = [in_path]
    else:
        files = list(find_las_files(in_path, args.recursive))
# ...existing code...

    print(f"找到 {len(files)} 个文件，输出目录：{out_dir}")
    converted = 0
    for f in tqdm(files):
        try:
            out_path, did = convert_file(f, out_dir, overwrite=args.overwrite, verbose=args.verbose)
            if did:
                converted += 1
        except Exception as e:
            print(f"转换失败：{f} -> {e}")

    print(f"完成：成功转换 {converted}/{len(files)} 个文件。")

if __name__ == "__main__":
    main()
# ...existing code...