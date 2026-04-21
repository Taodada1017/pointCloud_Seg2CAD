# -*- coding: utf-8 -*-
"""
GUI API适配器模块
连接lidar_gui.py中的GUI按钮与api_client.py中的接口
实现前后端分离，保持GUI和API代码的独立性
"""

import os
import sys
import numpy as np
from PyQt5.QtWidgets import QMessageBox, QProgressDialog, QLabel, QTextEdit, QGroupBox, QWidget, QVBoxLayout, QPlainTextEdit, QSizePolicy, QFrame, QDialog, QScrollArea, QApplication
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QEvent

# 导入API客户端
from api_client import api_client
# 导入线段编辑器组件
from line_editor_qt import TestWindow, LineSegment, LineSegments
# 导入图片预览组件
#from image_preview_widget import ImagePreviewWidget


class WorkerThread(QThread):
    """工作线程，用于在后台执行API调用，避免阻塞GUI"""
    
    # 定义信号
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, operation, *args, **kwargs):
        """初始化工作线程
        
        Args:
            operation: 要执行的API函数
            *args: API函数的位置参数
            **kwargs: API函数的关键字参数
        """
        super().__init__()
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        """执行API调用"""
        try:
            self.progress.emit("正在处理请求...")
            result = self.operation(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ImagePreviewWidget(QWidget):
    """图片预览控件，支持鼠标移动和滚轮缩放，用于嵌入到现有界面中"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path = None
        self.init_ui()
        self.setup_connections()
        # 添加深色背景样式
        self.setStyleSheet("background-color: #2c2c2c;")

    def init_ui(self):
        """初始化用户界面"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 禁用水平滚动条
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)    # 禁用垂直滚动条
        self.scroll_area.setFocusPolicy(Qt.WheelFocus)  # 确保滚轮事件能被正确捕获
        self.scroll_area.setWidgetResizable(False)  # 由我们手动控制尺寸
        
        # 创建图片容器
        self.image_container = QWidget()
        self.image_container.setMouseTracking(True)
        self.scroll_area.setWidget(self.image_container)
        
        # 为容器设置布局
        container_layout = QVBoxLayout(self.image_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建图片标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMouseTracking(True)
        self.image_label.setScaledContents(True)  # 确保图片填充整个标签
        container_layout.addWidget(self.image_label)
        
        # 添加滚动区域到主布局
        main_layout.addWidget(self.scroll_area)
        
        # 初始化缩放相关变量
        self.scale_factor = 1.0
        self.zoom_step = 0.1
        self.min_scale = 0.1
        self.max_scale = 5.0
        
        # 鼠标拖动相关变量
        self.is_dragging = False
        self.last_pos = None
        
        # 为滚动区域的视口添加事件过滤器
        self.scroll_area.viewport().installEventFilter(self)
        self.image_label.installEventFilter(self)
        
        # 确保控件可以接收键盘事件
        self.setFocusPolicy(Qt.WheelFocus)
    
    def load_image(self, image_path=None):
        """加载并显示图片
        
        Args:
            image_path: 可选，要加载的图片路径。如果为None，则使用当前已设置的路径
        """
        # 如果提供了新的图片路径，则更新
        if image_path is not None:
            self.image_path = image_path
        
        if not self.image_path or not os.path.exists(self.image_path):
            print(f"错误: 图片文件不存在: {self.image_path}")
            return False
        
        # 加载图片
        self.pixmap = QPixmap(self.image_path)
        if self.pixmap.isNull():
            print("错误: 无法加载图片")
            return False
        
        # 设置初始缩放因子为一个大于1.0的值，模拟已使用滚轮放大的效果
        # 这里设置为1.5倍，使图片加载时就已经被放大
        self.scale_factor = 1.8
        
        # 初始加载时自适应显示
        self.fit_to_view()
        
        # 更新滚动条策略
        self.update_scroll_policies()
        return True
        
    def fit_to_view(self):
        """将图片自适应显示到预览区域，确保完全撑满预览框"""
        if not hasattr(self, 'pixmap') or self.pixmap.isNull():
            return
        
        # 获取预览区域的可用尺寸（使用滚动区域视口的尺寸）
        viewport_size = self.scroll_area.viewport().size()
        # 如果视口尺寸过小，使用一个最小尺寸
        if viewport_size.width() < 100 or viewport_size.height() < 100:
            viewport_size = QSize(800, 600)  # 默认尺寸
        
        # 计算基础缩放因子，确保图片完全撑满预览框
        image_size = self.pixmap.size()
        width_ratio = viewport_size.width() / image_size.width()
        height_ratio = viewport_size.height() / image_size.height()
        base_scale = max(width_ratio, height_ratio)
        
        # 应用预先设置的放大倍数（在load_image中设置的self.scale_factor）
        # 这里我们将基础缩放因子乘以预先设置的缩放因子
        scale_factor = base_scale * self.scale_factor
        
        # 限制缩放范围
        scale_factor = max(self.min_scale, min(self.max_scale, scale_factor))
        
        # 应用缩放
        self.apply_scale(scale_factor)
    
    def reset_zoom(self):
        """重置缩放，显示原始大小"""
        self.apply_scale(1.0)
    
    def keyPressEvent(self, event):
        """处理键盘事件"""
        # 空格键重置缩放
        if event.key() == Qt.Key_Space:
            self.reset_zoom()
            return True
        # F键适应窗口
        elif event.key() == Qt.Key_F:
            self.fit_to_view()
            return True
        # 方向键移动
        elif event.key() in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
            h_bar = self.scroll_area.horizontalScrollBar()
            v_bar = self.scroll_area.verticalScrollBar()
            step = 20  # 滚动步长
            
            if event.key() == Qt.Key_Up:
                v_bar.setValue(v_bar.value() - step)
            elif event.key() == Qt.Key_Down:
                v_bar.setValue(v_bar.value() + step)
            elif event.key() == Qt.Key_Left:
                h_bar.setValue(h_bar.value() - step)
            elif event.key() == Qt.Key_Right:
                h_bar.setValue(h_bar.value() + step)
            return True
        
        return super().keyPressEvent(event)
    
    def setup_connections(self):
        """设置信号连接"""
        # 启用鼠标跟踪
        self.image_label.setMouseTracking(True)
        self.scroll_area.viewport().setMouseTracking(True)
    
    def wheelEvent(self, event):
        """处理鼠标滚轮事件，实现缩放"""
        # 获取滚轮方向
        delta = event.angleDelta().y()
        
        # 计算新的缩放因子
        if delta > 0:
            # 放大
            new_scale = self.scale_factor + self.zoom_step
        else:
            # 缩小
            new_scale = self.scale_factor - self.zoom_step
        
        # 限制缩放范围
        new_scale = max(self.min_scale, min(self.max_scale, new_scale))
        
        # 应用缩放
        self.apply_scale(new_scale)
    
    def update_scroll_policies(self):
        """根据图片大小和滚动区域大小更新滚动条策略"""
        scroll_size = self.scroll_area.viewport().size()
        image_size = self.image_container.size()
        
        # 如果图片宽度大于滚动区域，显示水平滚动条
        if image_size.width() > scroll_size.width():
            self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        else:
            self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 如果图片高度大于滚动区域，显示垂直滚动条
        if image_size.height() > scroll_size.height():
            self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        else:
            self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    
    def apply_scale(self, scale_factor):
        """应用缩放因子，基于原图尺寸进行缩放"""
        if not hasattr(self, 'pixmap') or self.pixmap.isNull():
            return
        
        self.scale_factor = scale_factor
        
        # 计算新的图片尺寸（基于原图大小乘以缩放因子）
        scaled_size = self.pixmap.size() * scale_factor
        
        # 缩放图片，保持宽高比
        scaled_pixmap = self.pixmap.scaled(
            scaled_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # 更新图片标签和容器
        self.image_label.setPixmap(scaled_pixmap)
        self.image_container.resize(scaled_size)
        
        # 更新滚动条策略
        self.update_scroll_policies()
        
        # 确保滚动区域更新
        self.scroll_area.update()
        self.scroll_area.viewport().update()
    
    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.last_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        """处理鼠标移动事件，实现拖动"""
        if self.is_dragging and self.last_pos:
            # 计算移动距离
            delta = event.pos() - self.last_pos
            
            # 移动滚动条
            h_bar = self.scroll_area.horizontalScrollBar()
            v_bar = self.scroll_area.verticalScrollBar()
            
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
            
            # 更新最后位置
            self.last_pos = event.pos()
    
    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            self.last_pos = None
    
    def eventFilter(self, obj, event):
        """事件过滤器，处理鼠标事件和滚轮事件"""
        # 处理滚轮事件
        if event.type() == QEvent.Wheel and obj == self.scroll_area.viewport():
            self.wheelEvent(event)
            return True
        
        # 处理鼠标事件
        if obj == self.image_label or obj == self.scroll_area.viewport():
            if event.type() == QEvent.MouseButtonPress:
                self.mousePressEvent(event)
                return True
            elif event.type() == QEvent.MouseMove:
                self.mouseMoveEvent(event)
                return True
            elif event.type() == QEvent.MouseButtonRelease:
                self.mouseReleaseEvent(event)
                return True
        
        return super().eventFilter(obj, event)
    
    def resizeEvent(self, event):
        """处理窗口调整事件"""
        super().resizeEvent(event)
        # 更新滚动条策略
        self.update_scroll_policies()

    def eventFilter(self, obj, event):
        """事件过滤器，处理鼠标事件和滚轮事件"""
        # 处理滚轮事件
        if event.type() == QEvent.Wheel and obj == self.scroll_area.viewport():
            self.wheelEvent(event)
            return True
        
        # 处理鼠标事件
        if obj == self.image_label or obj == self.scroll_area.viewport():
            if event.type() == QEvent.MouseButtonPress:
                self.mousePressEvent(event)
                return True
            elif event.type() == QEvent.MouseMove:
                self.mouseMoveEvent(event)
                return True
            elif event.type() == QEvent.MouseButtonRelease:
                self.mouseReleaseEvent(event)
                return True
        
        return super().eventFilter(obj, event)
    
    def resizeEvent(self, event):
        """处理窗口调整事件"""
        super().resizeEvent(event)
        # 更新滚动条策略
        self.update_scroll_policies()
    
    def showEvent(self, event):
        """处理窗口显示事件"""
        super().showEvent(event)
        # 更新滚动条策略
        self.update_scroll_policies()


class GUIApiAdapter:
    """GUI和API的适配器类"""
    
    def __init__(self, lidar_visualizer):
        """初始化适配器
        
        Args:
            lidar_visualizer: LidarVisualizer主窗口实例
        """
        self.lidar_visualizer = lidar_visualizer
        self.current_page = None
        self.worker = None
        self.progress_dialog = None
        self.connect_signals()
        self.pcd_file_path = None  #使用全局的路径变量，当点云预览设置了路径，这里也会更新
        self.config_file_path = None
    
    def connect_signals(self):
        """连接GUI信号和API处理函数"""
        # 连接页面切换信号，用于跟踪当前活动页面
        if hasattr(self.lidar_visualizer, 'switch_page'):
            # 如果有switch_page方法，我们需要确保页面切换时更新current_page
            # 这里我们需要一个hack来捕获页面切换事件
            original_switch_page = self.lidar_visualizer.switch_page
            
            def wrapped_switch_page(index):
                original_switch_page(index)
                self.current_page = self.lidar_visualizer.pages[index]
                self.connect_page_buttons(self.current_page)
            
            self.lidar_visualizer.switch_page = wrapped_switch_page
        
        # 初始化当前页面
        if hasattr(self.lidar_visualizer, 'pages') and self.lidar_visualizer.pages:
            self.current_page = self.lidar_visualizer.pages[0]
            self.connect_page_buttons(self.current_page)
    
    def connect_page_buttons(self, page):
        """连接特定页面的按钮信号
        
        Args:
            page: 要连接按钮的页面实例
        """
        print(f"开始连接页面按钮: {page.title if hasattr(page, 'title') else '未知页面'}")
        
        # 确保文件路径字典存在且正确初始化
        if not hasattr(page, '_file_paths') or not isinstance(page._file_paths, dict):
            page._file_paths = {'pcd': None, 'config': None}
            print("已初始化文件路径字典")
        
        # 确保字典包含所需的键
        required_keys = ['pcd', 'config']
        for key in required_keys:
            if key not in page._file_paths:
                page._file_paths[key] = None
                print(f"已添加缺失的键: {key}")
        
        # 定义包装后的文件选择函数
        def wrapped_select_pcd():
            """包装后的PCD文件选择方法，保存完整路径"""
            print("调用PCD文件选择方法")
            # 直接使用QFileDialog获取文件
            from PyQt5.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getOpenFileName(page, "选择PCD文件", "", "PCD Files (*.pcd)")
            if file_path and os.path.exists(file_path):
                # 保存完整路径
                page._file_paths['pcd'] = file_path
                print(f"已保存PCD文件路径: {file_path}")
                # 只显示文件名
                page.pcd_file_button.setText(os.path.basename(file_path))
                return True
            elif file_path:
                print(f"错误: 选择的PCD文件不存在: {file_path}")
            return False
        
        def wrapped_select_config():
            """包装后的配置文件选择方法，保存完整路径"""
            print("调用配置文件选择方法")
            # 直接使用QFileDialog获取配置文件
            from PyQt5.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getOpenFileName(page, "选择配置文件", "", "YAML Files (*.yaml)")
            if file_path and os.path.exists(file_path):
                # 保存完整路径
                page._file_paths['config'] = file_path
                print(f"已保存配置文件路径: {file_path}")
                # 只显示文件名
                page.config_file_button.setText(os.path.basename(file_path))
                return True
            elif file_path:
                print(f"错误: 选择的配置文件不存在: {file_path}")
            return False
        
        # 直接连接按钮的点击事件
        if hasattr(page, 'pcd_file_button'):
            # 先断开现有连接
            try:
                page.pcd_file_button.clicked.disconnect()
                print("已断开PCD按钮的现有连接")
            except (TypeError, RuntimeError):
                pass
            # 连接到我们的包装方法
            page.pcd_file_button.clicked.connect(wrapped_select_pcd)
            print("已直接连接PCD按钮到包装方法")
        
        if hasattr(page, 'config_file_button'):
            # 先断开现有连接
            try:
                page.config_file_button.clicked.disconnect()
                print("已断开配置按钮的现有连接")
            except (TypeError, RuntimeError):
                pass
            # 连接到我们的包装方法
            page.config_file_button.clicked.connect(wrapped_select_config)
            print("已直接连接配置按钮到包装方法")
        
        # 安全地连接生成按钮（只有当按钮存在时）
        if hasattr(page, 'generate_button'):
            # 先断开所有现有连接，避免重复连接
            try:
                page.generate_button.clicked.disconnect()
                print("已断开生成按钮的现有连接")
            except (TypeError, RuntimeError):
                # 如果没有连接，忽略错误
                pass
            
            # 根据页面标题连接到相应的API调用
            page_title = page.title if hasattr(page, 'title') else ''
            if "预览" in page_title:
                page.generate_button.clicked.connect(lambda: self.run_preview_api(page))
                print("已连接预览API到生成按钮")
            elif "投影" in page_title:
                page.generate_button.clicked.connect(lambda: self.run_projection_api(page))
                print("已连接投影API到生成按钮")
            elif "检测" in page_title:
                page.generate_button.clicked.connect(lambda: self.run_detection_api(page))
                print("已连接检测API到生成按钮")
        
        # 连接编辑线段按钮
        if hasattr(page, 'edit_button'):
            # 先断开所有现有连接
            try:
                page.edit_button.clicked.disconnect()
                print("已断开编辑线段按钮的现有连接")
            except (TypeError, RuntimeError):
                pass
            # 连接到我们的编辑线段方法
            page.edit_button.clicked.connect(lambda: self.run_line_edit_function(page))
            print("已连接编辑线段功能到编辑线段按钮")
        
        # 连接重置参数按钮
        if hasattr(page, 'reset_button'):
            # 先断开所有现有连接
            try:
                page.reset_button.clicked.disconnect()
                print("已断开重置参数按钮的现有连接")
            except (TypeError, RuntimeError):
                pass
            # 连接到我们的重置参数方法
            page.reset_button.clicked.connect(lambda: self.reset_page_parameters(page))
            print("已连接重置参数功能到重置参数按钮")
        
        print("页面按钮连接完成")
    
    def _check_api_status(self):
        """检查API服务是否运行
        
        Returns:
            bool: API服务是否正常运行
        """
        if not api_client.is_api_running():
            QMessageBox.critical(
                self.current_page,
                "API服务未运行",
                "请先启动API服务。\n可以通过运行 'python api.py' 来启动服务。"
            )
            return False
        return True
    
    def _get_file_path(self, page, file_type):
        """获取文件的完整路径
        
        Args:
            page: 页面实例
            file_type: 文件类型 ('pcd' 或 'config')
            
        Returns:
            str: 文件的完整路径，如果未选择则返回None
        """
        # 详细调试信息
        print(f"尝试获取{file_type}文件路径，页面: {page}")
        
        # 首先检查文件路径字典是否存在且包含正确的键
        if not hasattr(page, '_file_paths'):
            print(f"警告: 页面缺少_file_paths属性")
            # 初始化字典
            page._file_paths = {'pcd': None, 'config': None}
        
        # 检查是否存在指定的文件类型键
        if file_type not in page._file_paths:
            print(f"警告: _file_paths中缺少{file_type}键")
            page._file_paths[file_type] = None
        
        # 获取保存的文件路径
        file_path = page._file_paths[file_type]
        print(f"保存的{file_type}文件路径: {file_path}")
        
        # 验证文件是否存在
        if file_path and os.path.exists(file_path):
            print(f"获取到有效{file_type}文件路径: {file_path}")
            return file_path
        elif file_path:
            print(f"错误: {file_type}文件路径存在但文件不存在: {file_path}")
        else:
            print(f"错误: {file_type}文件路径为空")
        
        # 如果没有有效的文件路径，提示用户重新选择
        file_type_text = "PCD" if file_type == 'pcd' else "配置"
        button_name = 'pcd_file_button' if file_type == 'pcd' else 'config_file_button'
        
        # 检查按钮文本是否显示已选择文件
        if hasattr(page, button_name):
            button = getattr(page, button_name)
            default_text = "请选择PCD文件" if file_type == 'pcd' else "请选择yaml文件"
            
            if button.text() != default_text:
                # 虽然按钮显示已选择，但路径没有正确保存
                print(f"按钮显示已选择{file_type_text}文件，但路径无效或未保存")
                QMessageBox.warning(
                    page,
                    "文件路径问题",
                    f"{file_type_text}文件已选择但路径未正确保存。\n请重新选择文件。"
                )
        
        return None
    
    def reset_page_parameters(self, page):
        """重置页面参数
        
        Args:
            page: 页面实例
        """
        print(f"重置页面参数: {page.title if hasattr(page, 'title') else '未知页面'}")
        
        # 如果页面自身有reset_parameters方法，调用它
        if hasattr(page, 'reset_parameters'):
            try:
                page.reset_parameters()
                print("已调用页面自身的reset_parameters方法")
            except Exception as e:
                print(f"调用页面reset_parameters方法时出错: {str(e)}")
        
        # 重置文件路径字典
        if hasattr(page, '_file_paths'):
            page._file_paths = {'pcd': None, 'config': None}
            print("已重置文件路径字典")
        
        # 重置适配器中的全局路径变量,分为在不同的界面的重置按钮
        if "预览" in page.title:
            self.pcd_file_path = None
            self.config_file_path = None
        print("已重置适配器中的全局路径变量")
    
    def _get_cfg_path(self, page):
        """获取配置文件路径
        
        Args:
            page: 页面实例
            
        Returns:
            str: 配置文件路径
        """
        # 对于非预览页面，需要配置文件
        cfg_path = self._get_file_path(page, 'config')
        if not cfg_path:
            QMessageBox.warning(page, "配置文件未选择", "请先选择配置文件")
        return cfg_path
    
    def _start_worker(self, operation, *args, page=None, **kwargs):
        """启动工作线程执行API调用
        
        Args:
            operation: API函数
            *args: 位置参数
            page: 页面实例
            **kwargs: 关键字参数
        """
        if page is None:
            page = self.current_page
        
        # 创建进度对话框
        self.progress_dialog = QProgressDialog("正在处理...", "取消", 0, 0, page)
        
        # 根据操作类型设置不同的标题
        if operation == api_client.preview_pcd:
            self.progress_dialog.setWindowTitle("生成预览中")
        elif operation == api_client.project_pcd:
            self.progress_dialog.setWindowTitle("生成投影中")
        elif operation == api_client.detect_lines:
            self.progress_dialog.setWindowTitle("检测线段中")
        else:
            self.progress_dialog.setWindowTitle("处理中")
        
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setCancelButtonText("取消")
        self.progress_dialog.canceled.connect(self._cancel_worker)
        
        # 创建并启动工作线程
        self.worker = WorkerThread(operation, *args, **kwargs)
        self.worker.finished.connect(lambda result: self._on_worker_finished(result, page))
        self.worker.error.connect(self._on_worker_error)
        self.worker.progress.connect(self.progress_dialog.setLabelText)
        self.worker.start()
        
        # 显示进度对话框
        self.progress_dialog.exec_()
    
    def _cancel_worker(self):
        """取消工作线程"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        if self.progress_dialog:
            self.progress_dialog.close()
    
    def _on_worker_finished(self, result, page):
        """处理工作线程完成的结果
        
        Args:
            result: API返回的结果
            page: 页面实例
        """
        print(f"工作线程完成: 结果={result}, 页面={page}")
        
        # 关闭进度对话框
        if self.progress_dialog:
            self.progress_dialog.close()
            print("进度对话框已关闭")
        
        # 保存检测结果到页面，用于后续编辑线段功能
        page_title = page.title if hasattr(page, 'title') else ""
        if "检测" in page_title and result:
            print("保存检测结果到页面，用于编辑线段功能")
            page._last_detection_result = result
        
        try:
            # 获取图片路径和信息文本
            image_path = None
            info_text = f"状态: {result.get('message', '未知')}\n"
            
            print(f"正在提取结果信息: message={result.get('message')}")
            
            # 检查页面标题
            page_title = page.title if hasattr(page, 'title') else ""
            print(f"页面标题: {page_title}")
            
            # 创建一个更全面的图片路径查找逻辑
            image_paths_to_try = []
            
            # 添加所有可能的图片路径键
            common_image_keys = [
                'preview_png', 'once_png', 'line_image', 'segment_image', 
                'result_image', 'preview_image', 'image_path', 'output_image',
                'visualization', 'processed_image'
            ]
            
            # 先添加通用键
            for key in common_image_keys:
                if key in result:
                    image_paths_to_try.append(result[key])
            
            # 然后扫描整个结果字典，查找可能的图片路径
            for key, value in result.items():
                if isinstance(value, str) and value.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    if value not in image_paths_to_try:
                        image_paths_to_try.append(value)
            
            # 尝试找到有效的图片路径
            found_image = False
            for img_path in image_paths_to_try:
                if img_path and isinstance(img_path, str):
                    # 规范化路径
                    img_path = os.path.normpath(img_path)
                    
                    # 尝试直接路径
                    if os.path.exists(img_path):
                        print(f"找到有效图片路径: {img_path}")
                        image_path = img_path
                        found_image = True
                        break
                    # 尝试相对于当前工作目录的路径
                    elif os.path.exists(os.path.join(os.getcwd(), img_path)):
                        rel_path = os.path.join(os.getcwd(), img_path)
                        print(f"找到有效相对路径图片: {rel_path}")
                        image_path = rel_path
                        found_image = True
                        break
                    # 尝试相对于脚本目录的路径
                    elif os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), img_path)):
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        script_rel_path = os.path.join(script_dir, img_path)
                        print(f"找到有效脚本相对路径图片: {script_rel_path}")
                        image_path = script_rel_path
                        found_image = True
                        break
            
            if not found_image:
                print("结果中没有找到有效的图片路径")
                # 打印所有尝试过的路径用于调试
                print(f"尝试过的图片路径: {image_paths_to_try}")
            
            # 创建完整的信息字典，确保线段信息正确传递
            full_info_dict = {}
            if 'message' in result:
                full_info_dict['状态'] = result['message']
            
            # 添加统计信息
            if 'line_counts' in result:
                print(f"找到线段统计信息: {result['line_counts']}")
                info_text += "统计信息:\n"
                for key, value in result['line_counts'].items():
                    # 转换键名为更易读的格式
                    readable_key = self._get_readable_key(key)
                    info_text += f"  - {readable_key}: {value}\n"
                    # 同时添加到字典中用于传递给_update_info_display
                    full_info_dict[readable_key] = value
            elif 'lines' in result and isinstance(result['lines'], dict):
                print(f"找到线段统计信息(lines): {result['lines']}")
                info_text += "统计信息:\n"
                for key, value in result['lines'].items():
                    # 转换键名为更易读的格式
                    readable_key = self._get_readable_key(key)
                    info_text += f"  - {readable_key}: {value}\n"
                    # 同时添加到字典中用于传递给_update_info_display
                    full_info_dict[readable_key] = value
            else:
                print("未找到线段统计信息")
            
            # 添加输出目录信息
            if 'out_dir' in result:
                info_text += f"输出目录: {result['out_dir']}\n"
                full_info_dict['输出目录'] = result['out_dir']
                print(f"找到输出目录: {result['out_dir']}")
            
            # 更新结果显示
            if image_path:
                print("调用_update_image_display更新图片显示")
                self._update_image_display(page, image_path)
            else:
                print("未找到有效图片路径，跳过图片更新")
            
            print("调用_update_info_display更新信息显示")
            self._update_info_display(page, full_info_dict)  # 使用完整字典而不仅仅是文本
            
            # 对于检测页面，额外确保线段信息显示正确
            if "检测" in page_title and hasattr(page, 'children'):
                print("处理检测页面，确保线段信息正确显示")
                for child in page.children():
                    if isinstance(child, type(page)) and hasattr(child, 'children'):
                        for grand_child in child.children():
                            if isinstance(grand_child, QGroupBox) and grand_child.title() == "线段信息":
                                print("找到线段信息QGroupBox")
                                for great_grand_child in grand_child.children():
                                    if isinstance(great_grand_child, QTextEdit):
                                        # 提取线段数量信息
                                        line_count_text = "线段检测结果:\n"
                                        if 'line_counts' in result:
                                            for key, value in result['line_counts'].items():
                                                readable_key = self._get_readable_key(key)
                                                line_count_text += f"{readable_key}: {value}\n"
                                        elif 'lines' in result and isinstance(result['lines'], dict):
                                            for key, value in result['lines'].items():
                                                readable_key = self._get_readable_key(key)
                                                line_count_text += f"{readable_key}: {value}\n"
                                        else:
                                            line_count_text += "未获取到线段信息\n"
                                        print(f"设置线段信息文本: {line_count_text}")
                                        great_grand_child.setText(line_count_text)
                                        break
        except Exception as e:
            error_msg = str(e)
            print(f"处理结果时出错: {error_msg}")
            self._on_worker_error(error_msg)
    
    def _on_worker_error(self, error_msg):
        """处理工作线程错误"""
        # 关闭进度对话框
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
        
        # 显示错误信息
        error_message = f"处理过程中发生错误:\n{error_msg}"
        print(f"API调用错误: {error_msg}")  # 调试信息
        
        # 使用QMessageBox显示错误
        if self.current_page:
            QMessageBox.critical(
                self.current_page,
                "错误",
                error_message
            )
        else:
            # 如果没有current_page，创建一个临时的QMessageBox
            QMessageBox.critical(
                None,
                "错误",
                error_message
            )
        
        # 清理工作线程
        if hasattr(self, 'worker') and self.worker:
            if self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait()
            self.worker = None
    
    def _update_image_display(self, page, image_path):
        """更新图片显示
        
        Args:
            page: 页面实例
            image_path: 图片路径
        """
        print(f"更新图片显示，路径: {image_path}")
        
        # 检查图片路径是否有效
        if not image_path:
            print("错误: 图片路径为空")
            return
        
        if not os.path.exists(image_path):
            print(f"错误: 图片文件不存在: {image_path}")
            return
        
        # 保存图片路径到页面
        page._last_image_path = image_path
        
        # 创建pixmap
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print("错误: 无法加载图片")
            return
        
        print(f"图片尺寸: {pixmap.width()}x{pixmap.height()}")
        
        # 首先尝试在所有可能的控件中找到合适的显示区域
        target_widget = None
        display_methods = [
            lambda p: getattr(p, 'canvas_widget', None),
            lambda p: getattr(p, 'preview_widget', None),
            lambda p: next((w for w in p.findChildren(QWidget) if w.objectName() and 'preview' in w.objectName().lower()), None),
            lambda p: next((w for w in p.findChildren(QWidget) if w.objectName() and 'canvas' in w.objectName().lower()), None),
            lambda p: next((w for w in p.findChildren(QWidget) if w.objectName() and 'image' in w.objectName().lower()), None)
        ]
        
        for method in display_methods:
            widget = method(page)
            if widget:
                target_widget = widget
                print(f"找到目标控件: {target_widget}, 类型: {target_widget.__class__.__name__}")
                print(f"控件尺寸: {target_widget.width()}x{target_widget.height()}")
                break
        
        # 如果找到了目标控件
        if target_widget:
            # 设置尺寸策略
            target_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
            # 检查是否已经有ImagePreviewWidget
            existing_preview_widget = None
            for child in target_widget.findChildren(QWidget):
                if isinstance(child, ImagePreviewWidget):
                    existing_preview_widget = child
                    print(f"找到现有的ImagePreviewWidget: {child}")
                    break
            
            # 如果没有现有的ImagePreviewWidget，创建一个新的
            if not existing_preview_widget:
                # 检查是否有布局，如果没有就创建一个
                layout = target_widget.layout()
                if not layout:
                    layout = QVBoxLayout(target_widget)
                    layout.setContentsMargins(0, 0, 0, 0)
                    layout.setSpacing(0)
                    print("创建新的布局")
                else:
                    # 清除现有布局中的所有项
                    while layout.count() > 0:
                        item = layout.takeAt(0)
                        if item.widget():
                            item.widget().deleteLater()
                    print("使用现有布局并清空内容")
                
                # 创建新的ImagePreviewWidget
                preview_widget = ImagePreviewWidget(target_widget)
                preview_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                
                # 添加到布局
                layout.addWidget(preview_widget)
                print("创建新的ImagePreviewWidget并添加到目标控件")
                existing_preview_widget = preview_widget
            
            # 使用ImagePreviewWidget加载图片
            existing_preview_widget.load_image(image_path)
            print(f"已使用ImagePreviewWidget加载图片: {image_path}")
            
            # 确保控件可见
            target_widget.show()
            
            # 保存图片路径到页面
            page._last_image_path = image_path
            print("图片已成功显示在预览控件中")
            print(f"已关联图片显示控件与预览对话框: {image_path}")
        else:
            # 如果没有找到特定的显示控件，尝试查找页面上的QFrame或其他容器
            container_widgets = page.findChildren(QFrame)
            if container_widgets:
                target_widget = container_widgets[0]
                print(f"未找到专门的预览控件，使用第一个QFrame: {target_widget}")
                # 递归调用，尝试在这个容器中显示
                self._update_image_display(target_widget, image_path)
            else:
                print("未找到任何合适的图片显示控件")
                # 作为最后的尝试，直接在页面上显示
                image_label = QLabel(page)
                image_label.setObjectName("fallback_image_label")
                image_label.setAlignment(Qt.AlignCenter)
                image_label.setPixmap(pixmap.scaled(
                    page.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))
                image_label.show()
                print("在页面上直接显示图片作为后备方案")
    
    def _update_info_display(self, page, info_text):
        """更新信息显示
        
        Args:
            page: 页面实例
            info_text: 信息文本或字典
        """
        print(f"更新信息显示: {info_text}")
        
        # 首先尝试找到线段信息框
        segment_info_box = None
        
        # 查找名为"线段信息"的QGroupBox
        for child in page.findChildren(QGroupBox):
            if child.title() == "线段信息" or "线段" in child.title():
                segment_info_box = child
                print("找到线段信息框")
                break
        
        # 如果找不到，尝试查找名称中包含"info"的控件
        if not segment_info_box:
            for child in page.findChildren(QWidget):
                if hasattr(child, 'objectName') and 'info' in child.objectName().lower():
                    segment_info_box = child
                    print("找到可能的信息显示控件")
                    break
        
        # 如果找到了线段信息框，在其中查找文本编辑控件
        if segment_info_box:
            text_edit = None
            # 查找QTextEdit控件
            for child in segment_info_box.findChildren(QTextEdit):
                text_edit = child
                break
            
            # 如果没找到QTextEdit，尝试查找QPlainTextEdit
            if not text_edit:
                for child in segment_info_box.findChildren(QPlainTextEdit):
                    text_edit = child
                    break
            
            # 如果找到了文本编辑控件，更新内容
            if text_edit:
                print("找到文本编辑控件，更新信息")
                # 格式化信息
                formatted_text = ""
                if isinstance(info_text, dict):
                    for key, value in info_text.items():
                        # 使用_get_readable_key获取可读的键名
                        readable_key = self._get_readable_key(key)
                        formatted_text += f"{readable_key}: {value}\n"
                else:
                    formatted_text = str(info_text)
                
                # 清空现有内容并设置新内容
                text_edit.clear()
                text_edit.setPlainText(formatted_text)
                # 确保文本控件可见
                text_edit.show()
            else:
                # 如果没有找到文本编辑控件，创建一个新的
                print("未找到文本编辑控件，创建新的QTextEdit")
                text_edit = QTextEdit(segment_info_box)
                text_edit.setReadOnly(True)
                text_edit.setGeometry(10, 30, segment_info_box.width() - 20, segment_info_box.height() - 40)
                
                # 设置内容
                if isinstance(info_text, dict):
                    formatted_text = ""
                    for key, value in info_text.items():
                        readable_key = self._get_readable_key(key)
                        formatted_text += f"{readable_key}: {value}\n"
                else:
                    formatted_text = str(info_text)
                
                text_edit.setPlainText(formatted_text)
                text_edit.show()
        else:
            # 如果没有找到线段信息框，尝试直接在页面上查找文本编辑控件
            print("未找到线段信息框，尝试在页面上查找文本控件")
            text_edit = None
            
            # 查找可能的文本编辑控件
            for text_type in [QTextEdit, QPlainTextEdit, QLabel]:
                for child in page.findChildren(text_type):
                    if hasattr(child, 'objectName') and ('info' in child.objectName().lower() or 'text' in child.objectName().lower()):
                        text_edit = child
                        break
                if text_edit:
                    break
            
            if text_edit:
                print(f"找到文本控件: {text_edit}")
                # 根据控件类型设置内容
                if isinstance(text_edit, (QTextEdit, QPlainTextEdit)):
                    text_edit.clear()
                    if isinstance(info_text, dict):
                        formatted_text = ""
                        for key, value in info_text.items():
                            readable_key = self._get_readable_key(key)
                            formatted_text += f"{readable_key}: {value}\n"
                    else:
                        formatted_text = str(info_text)
                    text_edit.setPlainText(formatted_text)
                elif isinstance(text_edit, QLabel):
                    if isinstance(info_text, dict):
                        # 对于标签，只显示部分关键信息
                        key_to_display = next(iter(info_text.keys()), "")
                        text_edit.setText(f"{key_to_display}: {info_text.get(key_to_display, '')}")
                    else:
                        text_edit.setText(str(info_text))
        
        # 无论是否找到控件，都保存信息到页面
        page._last_info_text = info_text
    
    def _show_image_preview(self, image_path):
        """在右侧预览区域显示图片，支持缩放和拖动
        
        Args:
            image_path: 要预览的图片路径
        """
        try:
            print(f"在预览区域显示图片，路径: {image_path}")
            
            # 确保路径存在
            if not os.path.exists(image_path):
                # 尝试规范化路径
                normalized_path = os.path.normpath(image_path)
                if os.path.exists(normalized_path):
                    image_path = normalized_path
                else:
                    # 尝试查找文件（处理相对路径或其他路径问题）
                    search_paths = [
                        os.path.abspath(image_path),
                        os.path.join(os.getcwd(), image_path) if os.path.exists(os.getcwd()) else None,
                        os.path.join(os.path.dirname(__file__), image_path) if os.path.exists(os.path.dirname(__file__)) else None
                    ]
                    
                    # 过滤掉None值
                    search_paths = [p for p in search_paths if p is not None]
                    
                    found_path = None
                    for path in search_paths:
                        if os.path.exists(path):
                            found_path = path
                            break
                    
                    if not found_path:
                        try:
                            # 检查current_page是否存在且有效
                            if hasattr(self, 'current_page') and self.current_page is not None:
                                QMessageBox.warning(self.current_page, "错误", f"预览图片不存在: {image_path}")
                            else:
                                QMessageBox.warning(None, "错误", f"预览图片不存在: {image_path}")
                        except RuntimeError:
                            print(f"无法显示错误消息对话框")
                        return
                    image_path = found_path
            
            # 检查当前页面是否存在
            if not hasattr(self, 'current_page') or self.current_page is None:
                print("当前页面不存在，无法在预览区域显示图片")
                # 作为后备方案，使用对话框显示
                self._show_image_dialog(image_path)
                return
            
            # 查找预览区域控件
            target_widget = None
            preview_widgets = [
                # 优先查找canvas_widget
                getattr(self.current_page, 'canvas_widget', None),
                # 其次查找preview_panel
                getattr(self.current_page, 'preview_panel', None),
                # 然后查找任何名称包含preview的控件
                next((w for w in self.current_page.findChildren(QWidget) if w.objectName() and 'preview' in w.objectName().lower()), None),
                # 最后查找任何名称包含canvas的控件
                next((w for w in self.current_page.findChildren(QWidget) if w.objectName() and 'canvas' in w.objectName().lower()), None)
            ]
            
            # 找到第一个有效的预览控件
            for widget in preview_widgets:
                if widget is not None:
                    target_widget = widget
                    break
            
            if target_widget is not None:
                print(f"找到预览区域控件: {target_widget}")
                
                # 检查是否已经有ImagePreviewWidget实例
                existing_preview = None
                for child in target_widget.findChildren(ImagePreviewWidget):
                    existing_preview = child
                    break
                
                if existing_preview is not None:
                    # 如果已有预览控件，直接加载新图片
                    print("使用现有预览控件")
                    existing_preview.load_image(image_path)
                else:
                    # 清除目标控件中的现有内容
                    layout = target_widget.layout()
                    if layout:
                        while layout.count() > 0:
                            item = layout.takeAt(0)
                            if item.widget():
                                item.widget().deleteLater()
                    else:
                        # 创建新的布局
                        layout = QVBoxLayout(target_widget)
                        layout.setContentsMargins(0, 0, 0, 0)
                        layout.setSpacing(0)
                    
                    # 创建新的ImagePreviewWidget实例
                    print("创建新的预览控件")
                    preview_widget = ImagePreviewWidget(target_widget)
                    preview_widget.load_image(image_path)
                    
                    # 将预览控件添加到布局
                    layout.addWidget(preview_widget)
                    
                    # 设置预览控件的尺寸策略，使其充满整个预览区域
                    preview_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                
                # 保存图片路径到页面
                self.current_page._last_image_path = image_path
                print("图片已成功显示在预览区域")
            else:
                print("未找到预览区域控件，使用对话框显示")
                # 作为后备方案，使用对话框显示
                self._show_image_dialog(image_path)
                
        except RuntimeError as e:
            # 特殊处理Qt对象已被删除的错误
            if "wrapped C/C++ object" in str(e):
                print(f"警告: Qt对象已被删除: {str(e)}")
            else:
                raise
        except Exception as e:
            print(f"显示图片预览时出错: {str(e)}")
            try:
                if hasattr(self, 'current_page') and self.current_page is not None:
                    QMessageBox.warning(self.current_page, "错误", f"无法显示图片预览: {str(e)}")
                else:
                    QMessageBox.warning(None, "错误", f"无法显示图片预览: {str(e)}")
            except RuntimeError:
                print(f"无法显示错误消息对话框")
    
    def _show_image_dialog(self, image_path):
        """作为后备方案，使用ImagePreviewWidget显示图片
        
        Args:
            image_path: 要预览的图片路径
        """
        try:
            print(f"使用ImagePreviewWidget显示图片预览，路径: {image_path}")
            # 直接使用_show_image_preview方法，它已经能处理各种情况
            self._show_image_preview(image_path)
        except Exception as e:
            print(f"显示图片预览时出错: {str(e)}")
            try:
                if hasattr(self, 'current_page') and self.current_page is not None:
                    QMessageBox.warning(self.current_page, "错误", f"无法显示图片预览: {str(e)}")
                else:
                    QMessageBox.warning(None, "错误", f"无法显示图片预览: {str(e)}")
            except RuntimeError:
                print(f"无法显示错误消息对话框")
    
    def eventFilter(self, obj, event):
        """事件过滤器，捕获控件点击事件
        
        Args:
            obj: 事件源对象
            event: 事件对象
            
        Returns:
            bool: 是否处理了事件
        """
        # 检查是否是图片标签的点击事件
        if (isinstance(obj, QLabel) and 
            hasattr(obj, 'objectName') and 
            obj.objectName() == "result_image_label" and
            event.type() == QEvent.MouseButtonPress):
            # 如果标签有图片路径属性或关联的页面有最后一张图片路径
            if hasattr(obj, '_image_path') and obj._image_path:
                self._show_image_preview(obj._image_path)
                return True
            elif hasattr(self, 'current_page') and hasattr(self.current_page, '_last_image_path'):
                self._show_image_preview(self.current_page._last_image_path)
                return True
        
        return super().eventFilter(obj, event) if hasattr(super(), 'eventFilter') else False
    
    def _get_readable_key(self, key):
        """将API返回的键名转换为更易读的格式
        
        Args:
            key: 原始键名
            
        Returns:
            str: 易读的键名
        """
        key_map = {
            'once': '直接检测线段数',
            'long': '长线段数',
            'short': '短线段数',
            'final': '最终线段数',
            'num_points': '点云点数',
            'num_points_original': '原始点云点数',
            'num_points_downsampled': '下采样后点云点数'
        }
        return key_map.get(key, key)
    
    def _setup_image_label(self, label, pixmap, image_path):
        """设置图片标签
        
        Args:
            label: QLabel控件
            pixmap: 图片对象
            image_path: 图片路径
        """
        try:
            # 设置图片
            label.setPixmap(pixmap)
            
            # 设置尺寸策略
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            label.setScaledContents(True)
            
            # 设置鼠标样式为手型
            label.setCursor(Qt.PointingHandCursor)
            
            # 保存图片路径到标签
            label._image_path = image_path
            
            # 保存原始的mousePressEvent
            original_mouse_press = label.mousePressEvent
            
            # 定义新的鼠标按下事件处理函数
            def on_mouse_press(event):
                try:
                    # 调用原始的鼠标按下事件
                    if original_mouse_press:
                        original_mouse_press(event)
                    
                    # 显示图片预览
                    self._show_image_preview(label._image_path)
                except RuntimeError as e:
                    # 捕获对象已被删除的错误
                    if "wrapped C/C++ object of type" in str(e) and "has been deleted" in str(e):
                        print(f"警告: 尝试访问已删除的对象: {str(e)}")
                    else:
                        raise
            
            # 设置鼠标按下事件处理器
            label.mousePressEvent = on_mouse_press
        except RuntimeError as e:
            # 捕获对象已被删除的错误
            print(f"设置图片标签时出错: {str(e)}")
    
    def run_preview_api(self, page):
        """运行点云预览API
        
        Args:
            page: 页面实例
        """
        print("开始处理生成预览请求...")
        
        # 检查API状态
        if not self._check_api_status():
            return
        
        # 获取PCD文件路径
        pcd_path = self._get_file_path(page, 'pcd')
        self.pcd_file_path = pcd_path
        
        # 详细验证文件路径
        if not pcd_path:
            QMessageBox.warning(page, "文件路径缺失", "请选择有效的PCD文件！")
            print("PCD文件路径获取失败")
            return
        
        # 验证文件是否真实存在
        if not os.path.exists(pcd_path):
            QMessageBox.warning(page, "文件不存在", f"PCD文件不存在：{pcd_path}")
            print(f"PCD文件不存在：{pcd_path}")
            return
        
        print(f"PCD文件路径：{pcd_path}")

        cfg_path = self._get_file_path(page, 'config')
        self.cfg_file_path = cfg_path
        # 详细验证配置文件路径
        if not cfg_path:
            QMessageBox.warning(page, "文件路径缺失", "请选择有效的配置文件！")
            print("配置文件路径获取失败")
            return
        
        if not os.path.exists(cfg_path):
            QMessageBox.warning(page, "文件不存在", f"配置文件不存在：{cfg_path}")
            print(f"配置文件不存在：{cfg_path}")
            return
        
        print(f"配置文件路径：{cfg_path}")
        
        # 获取视角参数
        theta = 0.0
        phi = 0.0
        distance = 2.0
        background_color = 0.0
        voxel_size = 0.01
        
        # 尝试从页面获取参数控件的值
        if hasattr(page, 'param_spin_boxes_preview'):
            spin_boxes = page.param_spin_boxes_preview
            if len(spin_boxes) > 0:
                theta = spin_boxes[0].value()
            if len(spin_boxes) > 1:
                phi = spin_boxes[1].value()
            if len(spin_boxes) > 2:
                distance = spin_boxes[2].value()
            if len(spin_boxes) > 3:
                background_color = spin_boxes[3].value()
            if len(spin_boxes) > 4:
                voxel_size = spin_boxes[4].value()
        
        print(f"预览参数：theta={theta}, phi={phi}, distance={distance}, background_color={background_color}, voxel_size={voxel_size}")
        
        # 启动工作线程
        print("启动工作线程处理预览请求")
        self._start_worker(
            api_client.preview_pcd,
            pcd_path,
            theta=theta,
            phi=phi,
            distance=distance,
            background_color=background_color,
            voxel_size=voxel_size,
            page=page
        )
        print("预览请求已提交到工作线程")
        
    
    def run_projection_api(self, page):
        """运行点云投影API
        
        Args:
            page: 页面实例
        """
        print("开始处理点云投影请求...")
        
        # 检查API状态
        if not self._check_api_status():
            return
        
        # # 获取文件路径
        # pcd_path = self._get_file_path(page, 'pcd')
        pcd_path=self.pcd_file_path
        # 详细验证PCD文件路径
        if not pcd_path:
            QMessageBox.warning(page, "文件路径缺失", "请选择有效的PCD文件！")
            print("PCD文件路径获取失败")
            return
        
        if not os.path.exists(pcd_path):
            QMessageBox.warning(page, "文件不存在", f"PCD文件不存在：{pcd_path}")
            print(f"PCD文件不存在：{pcd_path}")
            return
        
        # 获取配置文件路径
        #cfg_path = self._get_file_path(page, 'config')
        cfg_path=self.cfg_file_path
        
        # 详细验证配置文件路径
        if not cfg_path:
            QMessageBox.warning(page, "文件路径缺失", "请选择有效的配置文件！")
            print("配置文件路径获取失败")
            return
        
        if not os.path.exists(cfg_path):
            QMessageBox.warning(page, "文件不存在", f"配置文件不存在：{cfg_path}")
            print(f"配置文件不存在：{cfg_path}")
            return
        
        print(f"PCD文件路径：{pcd_path}")
        print(f"配置文件路径：{cfg_path}")
        
        # 获取投影比例参数
        projection_scale = 30.0
        if hasattr(page, 'param_spin_boxes_projection'):
            spin_boxes = page.param_spin_boxes_projection
            if len(spin_boxes) > 0:
                projection_scale = spin_boxes[0].value()
        
        print(f"投影比例参数：{projection_scale}")
        
        # 启动工作线程
        print("启动工作线程处理投影请求")
        self._start_worker(
            api_client.project_pcd,
            pcd_path,
            cfg_path,
            save_png=True,
            projection_scale=projection_scale,
            page=page
        )
        print("投影请求已提交到工作线程")
    
    def run_detection_api(self, page):
        """运行线段检测API
        
        Args:
            page: 页面实例
        """
        print("开始处理线段检测请求...")
        
        # 检查API状态
        if not self._check_api_status():
            return
        
        # 获取文件路径
        #pcd_path = self._get_file_path(page, 'pcd')
        pcd_path=self.pcd_file_path
        
        # 详细验证PCD文件路径
        if not pcd_path:
            QMessageBox.warning(page, "文件路径缺失", "请选择有效的PCD文件！")
            print("PCD文件路径获取失败")
            return
        
        if not os.path.exists(pcd_path):
            QMessageBox.warning(page, "文件不存在", f"PCD文件不存在：{pcd_path}")
            print(f"PCD文件不存在：{pcd_path}")
            return
        
        # 获取配置文件路径
        #cfg_path = self._get_file_path(page, 'config')
        cfg_path=self.cfg_file_path
        
        # 详细验证配置文件路径
        if not cfg_path:
            QMessageBox.warning(page, "文件路径缺失", "请选择有效的配置文件！")
            print("配置文件路径获取失败")
            return
        
        if not os.path.exists(cfg_path):
            QMessageBox.warning(page, "文件不存在", f"配置文件不存在：{cfg_path}")
            print(f"配置文件不存在：{cfg_path}")
            return
        
        print(f"PCD文件路径：{pcd_path}")
        print(f"配置文件路径：{cfg_path}")
        
        # 获取最小线段长度参数
        min_line_length = 30
        if hasattr(page, 'param_spin_boxes_detect'):
            spin_boxes = page.param_spin_boxes_detect
            if len(spin_boxes) > 0:
                min_line_length = spin_boxes[0].value()
        
        print(f"最小线段长度参数：{min_line_length}")
        
        # 启动工作线程
        print("启动工作线程处理线段检测请求")
        self._start_worker(
            api_client.detect_lines,
            pcd_path,
            cfg_path,
            min_line_length=min_line_length,
            page=page,
        )
        print("线段检测请求已提交到工作线程")
        
    def run_line_edit_function(self, page):
        """运行线段编辑功能，打开TestWindow并传递数据
        
        Args:
            page: 页面实例
        """
        print("开始处理编辑线段请求...")
        
        # 检查页面是否有检测结果数据
        if not hasattr(page, '_last_detection_result'):
            QMessageBox.warning(page, "没有检测结果", "请先运行线段检测，然后再编辑线段！")
            print("页面没有保存的检测结果数据")
            return
        
        # 获取保存的检测结果
        detection_result = page._last_detection_result
        print(f"获取到保存的检测结果: {detection_result}")
        
        # 检查结果中是否有img和serialized_linesegs
        if 'img' not in detection_result or 'serialized_linesegs' not in detection_result:
            QMessageBox.warning(page, "数据不完整", "检测结果中缺少必要的数据！")
            print("检测结果中缺少img或serialized_linesegs数据")
            return
        
        try:
            # 转换img列表为numpy数组
            img_list = detection_result['img']
            img_array = np.array(img_list, dtype=np.uint8)
            print(f"成功转换img数据，形状: {img_array.shape}, 类型: {img_array.dtype}")
            
            # 转换serialized_linesegs为LineSegment对象
            serialized_linesegs = detection_result['serialized_linesegs']
            line_segments = []
            
            for seg_data in serialized_linesegs:
                # 确保每个线段数据包含所有必要的字段
                if isinstance(seg_data, dict):
                    # 同时支持两种格式：start/end 和 point_a/point_b
                    if 'start' in seg_data and 'end' in seg_data:
                        start = np.array(seg_data['start'], dtype=np.float32)
                        end = np.array(seg_data['end'], dtype=np.float32)
                    elif 'point_a' in seg_data and 'point_b' in seg_data:
                        start = np.array(seg_data['point_a'], dtype=np.float32)
                        end = np.array(seg_data['point_b'], dtype=np.float32)
                    else:
                        print(f"跳过无效的线段数据: {seg_data}")
                        continue
                    
                    # 创建LineSegment对象，只传递point_a和point_b参数
                    line_segment = LineSegment(start, end)
                    
                    # 如果有置信度数据，存储为额外属性
                    if 'confidence' in seg_data:
                        line_segment.confidence = seg_data['confidence']
                    
                    # 如果有索引数据，存储为额外属性
                    if 'index' in seg_data:
                        line_segment.index = seg_data['index']
                        print(f"创建线段: 索引={seg_data['index']}, 起点={start}, 终点={end}")
                    else:
                        print(f"创建线段: 起点={start}, 终点={end}")
                    
                    line_segments.append(line_segment)
                else:
                    print(f"跳过无效的线段数据: {seg_data}")
            
            # 创建LineSegments对象
            linesegs_obj = LineSegments(line_segments)
            print(f"成功创建LineSegments对象，包含{len(line_segments)}条线段")
            
            # 确保QApplication实例存在
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
                app.setStyle("Fusion")
                
                # 设置全局字体
                font = app.font()
                font.setFamily("SimHei")
                app.setFont(font)
            
            # 创建并显示TestWindow
            print("创建TestWindow实例并传递数据")
            # 将窗口保存为实例属性，确保不会被垃圾回收
            self.current_line_window = TestWindow(img_array, linesegs_obj)
            self.current_line_window.setWindowTitle("线段编辑器")
            self.current_line_window.show()
            
            print("线段编辑器已成功打开")
            
        except Exception as e:
            error_msg = f"打开线段编辑器时出错: {str(e)}"
            print(error_msg)
            QMessageBox.critical(page, "错误", error_msg)


def connect_gui_with_api(lidar_visualizer):
    """连接GUI和API的便捷函数
    
    Args:
        lidar_visualizer: LidarVisualizer主窗口实例
        
    Returns:
        GUIApiAdapter: 适配器实例
    """
    return GUIApiAdapter(lidar_visualizer)