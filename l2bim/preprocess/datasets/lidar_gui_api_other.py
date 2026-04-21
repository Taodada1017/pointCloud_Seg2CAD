# -*- coding: utf-8 -*-
"""
LiDAR点云处理GUI应用 - API集成版
此版本通过api_client模块调用后端API，实现前后端分离
"""

import sys
import os
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QGroupBox, QTabWidget, 
    QSplitter, QFrame, QMessageBox, QProgressDialog, QScrollArea
)
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 导入API客户端模块
from api_client import api_client
# 导入图片预览组件
from gui_api_adapter import ImagePreviewWidget


class WorkerThread(QThread):
    """工作线程，用于执行耗时的API调用操作"""
    
    # 定义信号
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, operation, *args, **kwargs):
        """初始化工作线程
        
        Args:
            operation: 要执行的操作函数
            *args: 传递给操作函数的位置参数
            **kwargs: 传递给操作函数的关键字参数
        """
        super().__init__()
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        """线程运行函数，执行API调用"""
        try:
            # 执行API调用
            self.progress.emit("正在处理请求...")
            result = self.operation(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainPage(QWidget):
    """主页面基类"""
    
    def __init__(self, parent=None):
        """初始化主页面"""
        super().__init__(parent)
        self.parent = parent
        self.file_path = None  # 存储点云文件路径
        self.config_path = None  # 存储配置文件路径
        self.setup_ui()
    
    def setup_ui(self):
        """设置UI界面"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        
        # 创建水平分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # 文件选择组
        file_group = QGroupBox("文件选择")
        file_layout = QVBoxLayout(file_group)
        
        # 点云文件选择按钮
        self.pcd_button = QPushButton("选择点云文件 (.pcd)")
        self.pcd_button.clicked.connect(self.select_pcd_file)
        self.pcd_path_label = QLabel("未选择文件")
        self.pcd_path_label.setWordWrap(True)
        file_layout.addWidget(self.pcd_button)
        file_layout.addWidget(self.pcd_path_label)
        
        # 配置文件选择按钮
        self.cfg_button = QPushButton("选择配置文件 (.yaml/.json)")
        self.cfg_button.clicked.connect(self.select_config_file)
        self.cfg_path_label = QLabel("未选择文件")
        self.cfg_path_label.setWordWrap(True)
        file_layout.addWidget(self.cfg_button)
        file_layout.addWidget(self.cfg_path_label)
        
        # 参数设置组
        param_group = QGroupBox("参数设置")
        param_layout = QVBoxLayout(param_group)
        self.setup_parameters(param_layout)
        
        # 按钮组
        button_group = QGroupBox("操作")
        button_layout = QVBoxLayout(button_group)
        
        # 运行按钮
        self.run_button = QPushButton("运行")
        self.run_button.setMinimumHeight(40)
        self.run_button.clicked.connect(self.run_operation)
        button_layout.addWidget(self.run_button)
        
        # 添加各个组到控制面板布局
        control_layout.addWidget(file_group)
        control_layout.addWidget(param_group)
        control_layout.addWidget(button_group)
        control_layout.addStretch()
        
        # 右侧结果显示区域
        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        
        # 结果预览组件
        self.preview_widget = ImagePreviewWidget()
        self.preview_widget.setMinimumHeight(400)
        self.preview_widget.setFrameShape(QFrame.Panel)
        self.preview_widget.setFrameShadow(QFrame.Sunken)
        
        # 结果信息区域
        self.info_label = QLabel("信息显示区域")
        self.info_label.setWordWrap(True)
        self.info_label.setFrameShape(QFrame.Panel)
        self.info_label.setFrameShadow(QFrame.Sunken)
        self.info_label.setMinimumHeight(100)
        
        # 添加到结果面板布局
        result_layout.addWidget(QLabel("预览结果:"))
        result_layout.addWidget(self.preview_widget)
        result_layout.addWidget(QLabel("处理信息:"))
        result_layout.addWidget(self.info_label)
        
        # 添加面板到分割器
        splitter.addWidget(control_panel)
        splitter.addWidget(result_panel)
        
        # 设置分割器比例
        splitter.setSizes([300, 700])
        
        # 添加分割器到主布局
        main_layout.addWidget(splitter)
    
    def setup_parameters(self, layout):
        """设置参数界面，子类可以重写此方法"""
        pass
    
    def select_pcd_file(self):
        """选择点云文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择点云文件", "", "PCD Files (*.pcd);;All Files (*)"
        )
        if file_path:
            self.file_path = file_path
            self.pcd_path_label.setText(os.path.basename(file_path))
    
    def select_config_file(self):
        """选择配置文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择配置文件", "", "Config Files (*.yaml *.json);;All Files (*)"
        )
        if file_path:
            self.config_path = file_path
            self.cfg_path_label.setText(os.path.basename(file_path))
    
    def run_operation(self):
        """运行操作，子类需要重写此方法"""
        pass
    
    def update_result(self, image_path, info_text):
        """更新结果显示
        
        Args:
            image_path: 结果图片路径
            info_text: 信息文本
        """
        # 显示图片
        if image_path and os.path.exists(image_path):
            # 使用预览组件加载图片
            self.preview_widget.load_image(image_path)
        
        # 显示信息
        self.info_label.setText(info_text)
    
    def show_error(self, message):
        """显示错误信息
        
        Args:
            message: 错误信息
        """
        QMessageBox.critical(self, "错误", message)
    
    def on_worker_finished(self, result):
        """工作线程完成时的处理，子类需要重写此方法
        
        Args:
            result: API返回的结果
        """
        pass
    
    def on_worker_error(self, error_msg):
        """工作线程错误时的处理
        
        Args:
            error_msg: 错误信息
        """
        self.progress_dialog.close()
        self.show_error(error_msg)
    
    def start_worker(self, operation, *args, **kwargs):
        """启动工作线程
        
        Args:
            operation: 要执行的操作函数
            *args: 传递给操作函数的位置参数
            **kwargs: 传递给操作函数的关键字参数
        """
        # 创建进度对话框
        self.progress_dialog = QProgressDialog("正在处理...", "取消", 0, 0, self)
        self.progress_dialog.setWindowTitle("请稍候")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setCancelButton(None)  # 不允许取消
        
        # 创建并启动工作线程
        self.worker = WorkerThread(operation, *args, **kwargs)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error.connect(self.on_worker_error)
        self.worker.progress.connect(self.progress_dialog.setLabelText)
        self.worker.start()
        
        # 显示进度对话框
        self.progress_dialog.exec_()


class PreviewPage(MainPage):
    """点云预览页面"""
    
    def __init__(self, parent=None):
        """初始化预览页面"""
        super().__init__(parent)
        self.setWindowTitle("点云预览")
    
    def setup_parameters(self, layout):
        """设置预览参数"""
        # 这里可以添加更多预览参数的设置，如视角、颜色等
        # 目前保持简单实现
        layout.addWidget(QLabel("预览参数（待实现）"))
    
    def run_operation(self):
        """运行点云预览操作"""
        # 检查是否选择了文件
        if not self.file_path:
            self.show_error("请先选择点云文件")
            return
        
        # 检查API是否可用
        if not api_client.is_api_running():
            self.show_error("API服务未运行，请先启动API服务")
            return
        
        # 启动工作线程调用API
        self.start_worker(
            api_client.preview_pcd,
            self.file_path,
            theta=0.0,
            phi=0.0,
            distance=2.0,
            background_color=0.0,
            voxel_size=0.01
        )
    
    def on_worker_finished(self, result):
        """处理预览结果"""
        self.progress_dialog.close()
        
        # 解析结果
        image_path = result.get("preview_png", "")
        info_text = f"消息: {result.get('message', '未知')}\n"
        
        # 添加统计信息
        line_counts = result.get("line_counts", {})
        if line_counts:
            for key, value in line_counts.items():
                info_text += f"{key}: {value}\n"
        
        # 更新结果显示
        self.update_result(image_path, info_text)


class ProjectionPage(MainPage):
    """点云投影页面"""
    
    def __init__(self, parent=None):
        """初始化投影页面"""
        super().__init__(parent)
        self.setWindowTitle("点云投影")
    
    def setup_parameters(self, layout):
        """设置投影参数"""
        layout.addWidget(QLabel("投影参数"))
    
    def run_operation(self):
        """运行点云投影操作"""
        # 检查是否选择了文件
        if not self.file_path:
            self.show_error("请先选择点云文件")
            return
        
        if not self.config_path:
            self.show_error("请先选择配置文件")
            return
        
        # 检查API是否可用
        if not api_client.is_api_running():
            self.show_error("API服务未运行，请先启动API服务")
            return
        
        # 启动工作线程调用API
        self.start_worker(
            api_client.project_pcd,
            self.file_path,
            self.config_path,
            save_png=True
        )
    
    def on_worker_finished(self, result):
        """处理投影结果"""
        self.progress_dialog.close()
        
        # 解析结果
        image_path = result.get("preview_png", "")
        info_text = f"消息: {result.get('message', '未知')}\n"
        
        # 添加统计信息
        line_counts = result.get("line_counts", {})
        if line_counts:
            for key, value in line_counts.items():
                info_text += f"{key}: {value}\n"
        
        # 更新结果显示
        self.update_result(image_path, info_text)


class DetectionPage(MainPage):
    """直接检测页面"""
    
    def __init__(self, parent=None):
        """初始化检测页面"""
        super().__init__(parent)
        self.setWindowTitle("线段检测")
        self.setup_detection_ui()
    
    def setup_detection_ui(self):
        """设置检测页面特有UI"""
        # 可以添加线段编辑按钮
        pass
    
    def setup_parameters(self, layout):
        """设置检测参数"""
        layout.addWidget(QLabel("检测参数"))
    
    def run_operation(self):
        """运行线段检测操作"""
        # 检查是否选择了文件
        if not self.file_path:
            self.show_error("请先选择点云文件")
            return
        
        if not self.config_path:
            self.show_error("请先选择配置文件")
            return
        
        # 检查API是否可用
        if not api_client.is_api_running():
            self.show_error("API服务未运行，请先启动API服务")
            return
        
        # 启动工作线程调用API
        self.start_worker(
            api_client.detect_lines,
            self.file_path,
            self.config_path
        )
    
    def on_worker_finished(self, result):
        """处理检测结果"""
        self.progress_dialog.close()
        
        # 解析结果
        image_path = result.get("once_png", "")
        info_text = f"消息: {result.get('message', '未知')}\n"
        
        # 添加统计信息
        line_counts = result.get("line_counts", {})
        if line_counts:
            for key, value in line_counts.items():
                info_text += f"{key}: {value}\n"
        
        # 更新结果显示
        self.update_result(image_path, info_text)


class LidarVisualizer(QMainWindow):
    """LiDAR可视化主窗口"""
    
    def __init__(self):
        """初始化主窗口"""
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        # 设置窗口标题和大小
        self.setWindowTitle("LiDAR点云处理系统 - API集成版")
        self.resize(1200, 800)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建选项卡控件
        self.tab_widget = QTabWidget()
        
        # 创建各个页面
        self.preview_page = PreviewPage(self)
        self.projection_page = ProjectionPage(self)
        self.detection_page = DetectionPage(self)
        
        # 添加页面到选项卡
        self.tab_widget.addTab(self.preview_page, "点云预览")
        self.tab_widget.addTab(self.projection_page, "点云投影")
        self.tab_widget.addTab(self.detection_page, "直接检测")
        
        # 添加选项卡到主布局
        main_layout.addWidget(self.tab_widget)
        
        # 设置深色主题
        self.set_dark_theme()
    
    def set_dark_theme(self):
        """设置深色主题"""
        # 创建深色样式表
        dark_style = """
        QWidget {
            background-color: #1e1e1e;
            color: #d4d4d4;
        }
        QGroupBox {
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            margin-top: 6px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding-left: 8px;
            padding-right: 8px;
            background-color: #252526;
        }
        QPushButton {
            background-color: #2d2d30;
            color: #d4d4d4;
            border: 1px solid #3c3c3c;
            border-radius: 3px;
            padding: 6px;
        }
        QPushButton:hover {
            background-color: #3a3a3d;
        }
        QPushButton:pressed {
            background-color: #0078d7;
        }
        QLabel {
            color: #d4d4d4;
        }
        QTabWidget::pane {
            border: 1px solid #3c3c3c;
            background-color: #1e1e1e;
        }
        QTabBar::tab {
            background-color: #2d2d30;
            color: #d4d4d4;
            padding: 8px 16px;
            border: 1px solid #3c3c3c;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #1e1e1e;
            border-bottom-color: #1e1e1e;
        }
        QTabBar::tab:hover {
            background-color: #3a3a3d;
        }
        QSplitter::handle {
            background-color: #3c3c3c;
            width: 6px;
        }
        QSplitter::handle:hover {
            background-color: #5a5a5a;
        }
        QProgressDialog {
            background-color: #1e1e1e;
            border: 1px solid #3c3c3c;
        }
        QMessageBox {
            background-color: #1e1e1e;
            color: #d4d4d4;
        }
        """
        
        # 应用样式表
        self.setStyleSheet(dark_style)


if __name__ == "__main__":
    # 创建应用程序实例
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle("Fusion")
    
    # 设置字体
    font = QFont()
    font.setFamily("Microsoft YaHei")
    app.setFont(font)
    
    # 创建并显示主窗口
    window = LidarVisualizer()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())