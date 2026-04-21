# -*- coding: utf-8 -*-
"""
点云预览与截图功能封装 v2
基于Open3D库实现点云可视化与截图功能
"""

import os
import sys
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image

# 让 Python 能从项目根目录 import preprocess.*
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def run_pcd_preview(
    pcd_path: str,
    theta: float = 0.0,
    phi: float = 0.0,
    distance: float = 2.0,
    background_color: float = 0.0,
    voxel_size: float = 0.01,
    save_dir: str = "./output/preview",
    output_filename: str = "preview_v2.png",
):
    """
    点云可视化与截图功能实现
    
    Args:
        pcd_path: 输入的点云文件路径 (.pcd/.ply)
        theta: 方位角，默认0.0
        phi: 仰角，默认0.0
        distance: 观察距离，默认2.0
        background_color: 背景色，默认0.0（黑色）
        voxel_size: 下采样体素大小，默认0.01
        save_dir: 输出目录
        output_filename: 输出文件名
        
    Returns:
        dict {
            "preview_path": 预览图像保存路径,
            "num_points_original": 原始点云点数,
            "num_points_downsampled": 下采样后的点云点数
        }
    """
    # 创建输出目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 读取点云文件
    print(f"[preview] 读取点云文件: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    # 检查点云是否成功读取
    num_points_original = len(pcd.points)
    if num_points_original == 0:
        raise ValueError(f"[preview] 无法读取点云或点云为空: {pcd_path}")
    print(f"[preview] 原始点云点数: {num_points_original}")
    
    # 应用下采样
    if voxel_size > 1e-6:  # 使用更精确的判断条件，避免浮点数精度问题
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
        num_points_downsampled = len(pcd_downsampled.points)
        print(f"[preview] 下采样后点云点数: {num_points_downsampled}")
        pcd_to_view = pcd_downsampled
    else:
        pcd_to_view = pcd
        num_points_downsampled = num_points_original
    
    # 设置可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.background_color = np.array([background_color, background_color, background_color])
    render_option.point_size = 2.0
    
    # 添加点云到可视化器
    vis.add_geometry(pcd_to_view)
    
    # 设置视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)  # 设置缩放
    ctr.set_lookat([0, 0, 0])  # 设置注视点
    ctr.set_up([0, 1, 0])  # 设置上方向
    
    # 根据方位角和仰角设置相机位置
    # 使用球坐标系计算相机位置
    x = distance * np.cos(phi) * np.sin(theta)
    y = distance * np.sin(phi)
    z = distance * np.cos(phi) * np.cos(theta)
    ctr.set_front([-x, -y, -z])  # 相机朝向（从相机指向原点的反方向）
    
    # 更新可视化器
    vis.update_geometry(pcd_to_view)
    vis.poll_events()
    vis.update_renderer()
    
    # 保存截图
    output_path = os.path.join(save_dir, output_filename)
    vis.capture_screen_image(output_path)
    print(f"[preview] 截图已保存至: {output_path}")
    
    # 关闭可视化器
    vis.destroy_window()
    
    return {
        "preview_path": output_path,
        "num_points_original": num_points_original,
        "num_points_downsampled": num_points_downsampled
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="点云预览与截图工具 v2")
    parser.add_argument("--pcd", required=True, help="输入点云文件路径 (.pcd/.ply)")
    parser.add_argument("--theta", type=float, default=0.0, help="方位角，默认0.0")
    parser.add_argument("--phi", type=float, default=0.0, help="仰角，默认0.0")
    parser.add_argument("--distance", type=float, default=2.0, help="观察距离，默认2.0")
    parser.add_argument("--background", type=float, default=0.0, help="背景色，默认0.0（黑色）")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="下采样体素大小，默认0.01")
    parser.add_argument("--out_dir", default="./output/preview", help="输出目录，默认./output/preview")
    parser.add_argument("--out_name", default="preview_v2.png", help="输出文件名，默认preview_v2.png")
    
    args = parser.parse_args()
    
    try:
        result = run_pcd_preview(
            pcd_path=args.pcd,
            theta=args.theta,
            phi=args.phi,
            distance=args.distance,
            background_color=args.background,
            voxel_size=args.voxel_size,
            save_dir=args.out_dir,
            output_filename=args.out_name
        )
        
        print("\n===== 点云预览完成 =====")
        print(f"预览图像路径: {result['preview_path']}")
        print(f"原始点云点数: {result['num_points_original']}")
        print(f"下采样后点数: {result['num_points_downsampled']}")
        
    except Exception as e:
        print(f"[错误] 预览过程中发生错误: {e}")
        sys.exit(1)