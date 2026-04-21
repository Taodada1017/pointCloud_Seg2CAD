# -*- coding: utf-8 -*-
"""
API客户端模块 - 封装所有与后端API的交互逻辑
实现前后端分离，GUI通过此模块调用后端API
"""

import os
import requests
from typing import Dict, Optional, Any


class ApiClient:
    """API客户端类，负责与后端API的所有交互"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        """初始化API客户端
        
        Args:
            base_url: API服务的基础URL，默认为http://127.0.0.1:8000
        """
        self.base_url = base_url
        self.session = requests.Session()
        # 设置请求超时时间
        self.timeout = 300  # 5分钟，考虑到点云处理可能需要较长时间
    
    def preview_pcd(self, pcd_path: str, theta: float = 0.0, phi: float = 0.0, 
                    distance: float = 2.0, background_color: float = 0.0, 
                    voxel_size: float = 0.01) -> Dict[str, Any]:
        """调用点云预览接口
        
        Args:
            pcd_path: 点云文件路径
            theta: 方位角
            phi: 仰角
            distance: 观察距离
            background_color: 背景色
            voxel_size: 下采样体素大小
            
        Returns:
            包含预览结果的字典，格式如下：
            {
                "message": str,  # 状态消息
                "preview_png": str,  # 预览图片路径
                "line_counts": dict,  # 点云点数统计
                "out_dir": str  # 输出目录
            }
            
        Raises:
            Exception: API调用失败时抛出异常
        """
        url = f"{self.base_url}/preview/v2"
        
        # 准备表单数据
        data = {
            "theta": str(theta),
            "phi": str(phi),
            "distance": str(distance),
            "background_color": str(background_color),
            "voxel_size": str(voxel_size)
        }
        
        # 准备文件数据
        with open(pcd_path, 'rb') as f:
            files = {'file': (os.path.basename(pcd_path), f, 'application/octet-stream')}
            
            try:
                response = self.session.post(url, data=data, files=files, timeout=self.timeout)
                response.raise_for_status()  # 检查响应状态
                return response.json()
            except requests.exceptions.RequestException as e:
                raise Exception(f"点云预览API调用失败: {str(e)}")
    
    def project_pcd(self, pcd_path: str, cfg_path: str, save_png: bool = True, projection_scale: float = 30.0) -> Dict[str, Any]:
        """调用点云投影接口
        
        Args:
            pcd_path: 点云文件路径
            cfg_path: 配置文件路径
            save_png: 是否保存PNG图片
            
        Returns:
            包含投影结果的字典，格式如下：
            {
                "message": str,  # 状态消息
                "preview_png": str,  # 投影图片路径
                "line_counts": dict,  # 点云点数统计
                "out_dir": str  # 输出目录
            }
            
        Raises:
            Exception: API调用失败时抛出异常
        """
        url = f"{self.base_url}/projection"
        
        # 准备表单数据
        data = {
            "cfg": cfg_path,
            "save_png": str(save_png).lower(),
            "projection_scale": str(projection_scale)
        }
        
        # 准备文件数据
        with open(pcd_path, 'rb') as f:
            files = {'file': (os.path.basename(pcd_path), f, 'application/octet-stream')}
            
            try:
                response = self.session.post(url, data=data, files=files, timeout=self.timeout)
                response.raise_for_status()  # 检查响应状态
                return response.json()
            except requests.exceptions.RequestException as e:
                raise Exception(f"点云投影API调用失败: {str(e)}")
    
    def detect_lines(self, pcd_path: str, cfg_path: str, min_line_length: int = 30) -> Dict[str, Any]:
        """调用线段检测接口
        
        Args:
            pcd_path: 点云文件路径
            cfg_path: 配置文件路径
            min_line_length: 最小线段长度参数
            
        Returns:
            包含检测结果的字典，格式如下：
            {
                "message": str,  # 状态消息
                "once_png": str,  # 检测结果图片路径
                "line_counts": dict,  # 线段数量统计
                "out_dir": str  # 输出目录
            }
            
        Raises:
            Exception: API调用失败时抛出异常
        """
        url = f"{self.base_url}/detect/once"
        
        # 准备表单数据
        data = {
            "cfg": cfg_path,
            "return_numpy": "true",
            "return_linesegs": "true",
            "min_line_length": str(min_line_length)
        }
        
        # 准备文件数据
        with open(pcd_path, 'rb') as f:
            files = {'file': (os.path.basename(pcd_path), f, 'application/octet-stream')}
            
            try:
                response = self.session.post(url, data=data, files=files, timeout=self.timeout)
                response.raise_for_status()  # 检查响应状态
                result = response.json()
                
                # 确保返回格式兼容
                # 如果响应中有serialized_linesegs但没有linesegs，添加linesegs引用指向serialized_linesegs
                if 'serialized_linesegs' in result and 'linesegs' not in result:
                    result['linesegs'] = result['serialized_linesegs']
                    
                return result
            except requests.exceptions.RequestException as e:
                raise Exception(f"线段检测API调用失败: {str(e)}")
    
    def is_api_running(self) -> bool:
        """检查API服务是否正常运行
        
        Returns:
            bool: API服务是否正常运行
        """
        try:
            response = self.session.get(f"{self.base_url}/docs", timeout=5)
            return response.status_code == 200
        except:
            return False


# 创建全局API客户端实例，方便导入使用
api_client = ApiClient()