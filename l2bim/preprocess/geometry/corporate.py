import open3d as o3d
import numpy as np
# 选一个 PCD 文件路径
pcd_path = r"F:\Google\Download\4F_Region3\4F_Region3\submap\data\000016.pcd"

# 读取点云
pcd = o3d.io.read_point_cloud(pcd_path)

# 打印基本信息
print(pcd)
print(np.asarray(pcd.points))  # 点坐标

# 可视化
o3d.visualization.draw_geometries([pcd], window_name="Submap Viewer")
