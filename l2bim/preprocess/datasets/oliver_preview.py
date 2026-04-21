import open3d as o3d
path=r"D:\work\l2bim\pcds\004.pcd"
pcd=o3d.io.read_point_cloud(path)
o3d.visualization.draw_geometries([pcd])