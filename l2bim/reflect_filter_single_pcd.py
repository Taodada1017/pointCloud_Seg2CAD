import argparse, numpy as np, open3d as o3d

def main():
    ap = argparse.ArgumentParser("Single-PCD reflection suppression (heuristic)")
    ap.add_argument("--pcd", required=True)
    ap.add_argument("--out_pcd", required=True)
    ap.add_argument("--voxel", type=float, default=0.05)     # 用于估计法向/密度
    ap.add_argument("--nb_neighbors", type=int, default=24)  # 统计滤波K
    ap.add_argument("--std_ratio", type=float, default=1.5)
    ap.add_argument("--planarity_th", type=float, default=0.35) # 越小越像平面(λ3/λ2)
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(args.pcd)

    # 1) 统计离群滤波（先粗滤）
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=args.nb_neighbors,
                                              std_ratio=args.std_ratio)

    # 2) 估计法向与“平面性”（基于特征值）
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=3*args.voxel, max_nn=50))
    # 近邻协方差的最小特征值/次小特征值 近似平面性
    k = 30
    pts = np.asarray(pcd.points)
    tree = o3d.geometry.KDTreeFlann(pcd)
    planar_mask = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        _, idx, _ = tree.search_knn_vector_3d(pcd.points[i], k)
        Q = pts[idx] - pts[idx].mean(0)
        cov = Q.T @ Q / max(len(idx)-1, 1)
        w, _ = np.linalg.eig(cov)
        w = np.sort(np.real(w))
        if w[1] < 1e-9: 
            planar = True
        else:
            planar = (w[0]/w[1]) < args.planarity_th
        planar_mask[i] = planar

    pcd = pcd.select_by_index(np.where(planar_mask)[0])

    # 3) 小簇/细长簇清理（镜面喷射多为小而细）
    labels = np.array(pcd.cluster_dbscan(eps=0.12, min_points=15, print_progress=False))
    keep_idx = []
    for lb in np.unique(labels):
        if lb < 0: 
            continue
        idx = np.where(labels==lb)[0]
        if len(idx) < 80:         # 小簇丢
            continue
        keep_idx.extend(idx)
    pcd = pcd.select_by_index(keep_idx)

    o3d.io.write_point_cloud(args.out_pcd, pcd, write_ascii=False, compressed=True)
    print(f"[OK] 写出：{args.out_pcd}  点数={len(np.asarray(pcd.points))}")

if __name__ == "__main__":
    main()
