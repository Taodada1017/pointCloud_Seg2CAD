import os
import glob
import argparse
import numpy as np
import open3d as o3d

def natural_key(s):
    # 让 0002.pcd < 0010.pcd
    return [int(t) if t.isdigit() else t for t in
            [u for u in ''.join([c if c.isdigit() else ' ' for c in os.path.basename(s)]).split()]]

def voxel_union(pc, voxel):
    """体素级去重：把点云量化到体素中心，合并重复点。"""
    if len(pc.points) == 0:
        return pc
    xyz = np.asarray(pc.points)
    if voxel > 0:
        q = np.round(xyz / voxel) * voxel
        # 唯一化
        uq, idx = np.unique(q, axis=0, return_index=True)
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(uq))
    return pc

def register_icp(source_ds, target_ds, thresh):
    # 点到平面 ICP 更稳（需要法向）
    result = o3d.pipelines.registration.registration_icp(
        source_ds, target_ds, thresh,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    return result.transformation

def main(args):
    files = sorted(glob.glob(os.path.join(args.in_dir, "*.pcd")))
    if len(files) == 0:
        print(f"[ERR] No .pcd in {args.in_dir}")
        return

    print(f"[INFO] Found {len(files)} frames.")
    # 读第一帧，作为全局坐标系
    base = o3d.io.read_point_cloud(files[0])
    if args.pre_voxel > 0:
        base_ds = base.voxel_down_sample(args.pre_voxel)
    else:
        base_ds = base

    base_ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=args.pre_voxel*3 if args.pre_voxel>0 else 0.3, max_nn=30))
    merged = base  # 用原始高密度版本作融合
    merged = voxel_union(merged, args.voxel)  # 先去一次重

    # 逐帧处理
    T_global = np.eye(4)
    for i, f in enumerate(files[1:], start=1):
        print(f"[RUN] {i:05d} <- {os.path.basename(f)}")
        src = o3d.io.read_point_cloud(f)

        # 低模用于配准
        src_ds = src.voxel_down_sample(args.pre_voxel) if args.pre_voxel>0 else src
        src_ds.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=args.pre_voxel*3 if args.pre_voxel>0 else 0.3, max_nn=30))

        if args.assume_aligned:
            T = np.eye(4)
        else:
            T = register_icp(src_ds, base_ds, args.icp_th)
        # 变换原始高密点云
        src.transform(T)

        # 融合 + 去重
        merged += src
        merged = voxel_union(merged, args.voxel)

        # 可选：更新目标为当前合并后的低模，提高鲁棒性
        base_ds = merged.voxel_down_sample(args.pre_voxel if args.pre_voxel>0 else 0.1)
        base_ds.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=args.pre_voxel*3 if args.pre_voxel>0 else 0.3, max_nn=30))

        print(f"   merged size ~ {len(merged.points):,}")

    # 导出
    os.makedirs(os.path.dirname(args.out_pcd), exist_ok=True)
    o3d.io.write_point_cloud(args.out_pcd, merged)
    print(f"[OK] saved -> {args.out_pcd}  (points={len(merged.points):,})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="子地图帧目录，例如 F:/.../submap/data")
    ap.add_argument("--out_pcd", default="merged_submap.pcd")
    ap.add_argument("--assume_aligned", action="store_true",
                    help="如果帧已在同一坐标系，勾上此项（跳过ICP）")
    ap.add_argument("--pre_voxel", type=float, default=0.08,
                    help="配准前的降采样体素（m），0.05~0.1常用")
    ap.add_argument("--voxel", type=float, default=0.05,
                    help="最终融合去重体素（m），决定地图分辨率与大小")
    ap.add_argument("--icp_th", type=float, default=0.4,
                    help="ICP匹配阈值（m），0.2~0.6常用")
    args = ap.parse_args()
    main(args)
