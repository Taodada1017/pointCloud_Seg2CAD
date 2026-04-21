# preprocess/geometry/run_submap_batch.py
import os, sys, glob
import numpy as np
sys.path.append(os.path.dirname(__file__))

import open3d as o3d
from lineseg import LineSegmentExtractor, LineSegments   # 提线段
from corner  import Corner                               # 角点对象

# ======== 配置：把这个目录改成你的 submap/data 路径 ========
SUBMAP_DIR = r"F:\Google\Download\4F_Region3\4F_Region3\submap\data"

# 输出目录（自动创建）
OUT_LINES_DIR   = os.path.join(SUBMAP_DIR, "out_lines")
OUT_CORNERS_DIR = os.path.join(SUBMAP_DIR, "out_corners")
os.makedirs(OUT_LINES_DIR,   exist_ok=True)
os.makedirs(OUT_CORNERS_DIR, exist_ok=True)

def process_one_pcd(pcd_path: str, voxel_size=0.10, flat=True, vis=False):
    """
    跑单帧：读 -> (可选)下采样 -> (可选)压平 -> 提线 -> 角点 -> 存结果
    """
    name = os.path.splitext(os.path.basename(pcd_path))[0]  # 000000
    print(f"[RUN] {name}  <-  {pcd_path}")

    # === 读点云 ===
    pcd = o3d.io.read_point_cloud(pcd_path)
    if len(pcd.points) == 0:
        print(f"  [WARN] 空点云，跳过")
        return False
    pts = np.asarray(pcd.points)
    print("Z-range:", float(pts[:,2].min()), float(pts[:,2].max()))
    mask = (pts[:,2] >= cfg.minHeight) & (pts[:,2] <= cfg.maxHeight)   # 名称按你的config为准
    print("Slice kept:", int(mask.sum()), "/", len(pts))
    # === 轻量预处理 ===
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))

    if flat:
        pts = np.asarray(pcd.points)
        pts[:, 2] = 0.0
        pcd.points = o3d.utility.Vector3dVector(pts)

    # 确保有颜色（提线器支持按颜色/聚类）
    if not pcd.has_colors():
        pts = np.asarray(pcd.points)
        colors = np.zeros_like(pts)
        colors[:] = np.array([0.0, 0.651, 0.929])
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # === 提线段 ===
    extractor = LineSegmentExtractor(walls_2d=pcd)
    extractor.extract_lineseg()
    lines: LineSegments = extractor.linesegs
    print(f"  lines:   {len(lines)}")

    # === 角点 ===
    corners = lines.intersections(threshold=0.10)
    print(f"  corners: {len(corners)}")

    # === 存结果 ===
    lines_out   = os.path.join(OUT_LINES_DIR,   f"lines_{name}.txt")
    corners_out = os.path.join(OUT_CORNERS_DIR, f"corners_{name}.txt")

    with open(lines_out, "w") as f:
        for seg in lines:
            a = seg.point_a; b = seg.point_b
            f.write(f"{a[0]} {a[1]} {b[0]} {b[1]}\n")

    with open(corners_out, "w") as f:
        for c in corners:
            x, y = c.get_xy()
            f.write(f"{x} {y}\n")

    print(f"  saved -> {os.path.relpath(lines_out, SUBMAP_DIR)}  |  {os.path.relpath(corners_out, SUBMAP_DIR)}")

    # 可视化（批量时默认关闭，单步调试可开）
    if vis:
        ls = lines.get_o3d_lineset()
        corner_boxes = [c.to_o3d(color=[1,0,0], radius=0.15) for c in corners]
        o3d.visualization.draw_geometries([pcd, ls] + corner_boxes)

    return True

def main():
    pcd_list = sorted(glob.glob(os.path.join(SUBMAP_DIR, "*.pcd")))
    if not pcd_list:
        print(f"[ERR] 在 {SUBMAP_DIR} 没找到 .pcd")
        return

    ok, fail = 0, 0
    for p in pcd_list:
        try:
            ok += 1 if process_one_pcd(p, voxel_size=0.10, flat=True, vis=False) else 0
        except Exception as e:
            print(f"  [FAIL] {p}  ->  {e}")
            fail += 1

    print(f"\n[SUMMARY] done. success={ok}, fail={fail}, total={len(pcd_list)}")
    print(f"results in:\n  {OUT_LINES_DIR}\n  {OUT_CORNERS_DIR}")

if __name__ == "__main__":
    main()
