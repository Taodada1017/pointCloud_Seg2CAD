#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
点云检测线段可视化界面
用于调用后端API进行点云预览、一次检测和分步检测
"""
import sys
import os
import requests
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QTabWidget,
    QGroupBox, QGridLayout, QTextEdit, QProgressBar
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class APICallThread(QThread):
    """
    API调用线程，避免阻塞主线程
    """
    finished = pyqtSignal(dict, str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, api_url, file_path, api_type):
        super().__init__()
        self.api_url = api_url
        self.file_path = file_path
        self.api_type = api_type

    def run(self):
        try:
            self.progress.emit(20)
            # 检查文件是否存在
            if not os.path.exists(self.file_path):
                self.error.emit("文件不存在")
                return

            self.progress.emit(40)
            # 准备文件数据
            files = {'file': open(self.file_path, 'rb')}
            
            self.progress.emit(60)
            # 发送请求
            response = requests.post(self.api_url, files=files)
            
            self.progress.emit(80)
            # 关闭文件
            files['file'].close()
            
            # 检查响应状态
            if response.status_code == 200:
                self.progress.emit(100)
                self.finished.emit(response.json(), self.api_type)
            else:
                self.error.emit(f"API调用失败: {response.status_code}")
        except Exception as e:
            self.error.emit(f"发生错误: {str(e)}")


class PCDVisualizer(QMainWindow):
    """
    点云可视化主窗口
    """
    def __init__(self):
        super().__init__()
        # API配置
        self.api_base_url = "http://127.0.0.1:8000"
        self.api_endpoints = {
            "preview": f"{self.api_base_url}/preview",
            "once": f"{self.api_base_url}/detect/once",
            "multi": f"{self.api_base_url}/detect/multi"
        }
        
        # 当前选择的文件
        self.selected_file = None
        
        # 初始化UI
        self.init_ui()

    def init_ui(self):
        """
        初始化用户界面
        """
        # 设置窗口标题和大小
        self.setWindowTitle("LiDAR点云检测系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 文件选择区域
        file_group = QGroupBox("文件选择")
        file_layout = QHBoxLayout()
        
        self.file_label = QLabel("未选择文件")
        self.file_label.setWordWrap(True)
        self.file_label.setMinimumWidth(400)
        
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_file)
        
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(browse_btn)
        file_group.setLayout(file_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        
        # 功能按钮区域
        button_group = QGroupBox("功能操作")
        button_layout = QGridLayout()
        
        self.preview_btn = QPushButton("点云预览")
        self.preview_btn.clicked.connect(lambda: self.call_api("preview"))
        self.preview_btn.setEnabled(False)
        
        self.once_detect_btn = QPushButton("一次检测")
        self.once_detect_btn.clicked.connect(lambda: self.call_api("once"))
        self.once_detect_btn.setEnabled(False)
        
        self.multi_detect_btn = QPushButton("分步检测")
        self.multi_detect_btn.clicked.connect(lambda: self.call_api("multi"))
        self.multi_detect_btn.setEnabled(False)
        
        button_layout.addWidget(self.preview_btn, 0, 0)
        button_layout.addWidget(self.once_detect_btn, 0, 1)
        button_layout.addWidget(self.multi_detect_btn, 0, 2)
        button_group.setLayout(button_layout)
        
        # 结果展示区域
        self.tab_widget = QTabWidget()
        
        # 预览结果标签页
        self.preview_tab = QWidget()
        self.preview_layout = QVBoxLayout(self.preview_tab)
        self.preview_image_label = QLabel("预览图像将显示在这里")
        self.preview_image_label.setAlignment(Qt.AlignCenter)
        self.preview_image_label.setStyleSheet("border: 1px solid #ccc;")
        self.preview_image_label.setMinimumHeight(400)  # 增加最小高度
        self.preview_image_label.setCursor(Qt.PointingHandCursor)  # 设置鼠标指针
        self.preview_image_label.mouseDoubleClickEvent = lambda event: self.show_full_image(event, self.preview_image_label)
        self.preview_layout.addWidget(self.preview_image_label)
        self.tab_widget.addTab(self.preview_tab, "预览结果")
        
        # 一次检测标签页
        self.once_tab = QWidget()
        self.once_layout = QVBoxLayout(self.once_tab)
        self.once_image_label = QLabel("一次检测结果将显示在这里")
        self.once_image_label.setAlignment(Qt.AlignCenter)
        self.once_image_label.setStyleSheet("border: 1px solid #ccc;")
        self.once_image_label.setMinimumHeight(350)  # 增加最小高度
        self.once_image_label.setCursor(Qt.PointingHandCursor)  # 设置鼠标指针
        self.once_image_label.mouseDoubleClickEvent = lambda event: self.show_full_image(event, self.once_image_label)
        self.once_text_edit = QTextEdit()
        self.once_text_edit.setReadOnly(True)
        self.once_text_edit.setMinimumHeight(100)
        self.once_layout.addWidget(self.once_image_label)
        self.once_layout.addWidget(self.once_text_edit)
        self.tab_widget.addTab(self.once_tab, "一次检测")
        
        # 分步检测标签页
        self.multi_tab = QWidget()
        self.multi_layout = QVBoxLayout(self.multi_tab)
        
        # 分步检测图像布局
        self.multi_images_layout = QVBoxLayout()
        
        self.multi_stage1_label = QLabel("阶段1: 长线提取结果将显示在这里")
        self.multi_stage1_label.setAlignment(Qt.AlignCenter)
        self.multi_stage1_label.setStyleSheet("border: 1px solid #ccc;")
        self.multi_stage1_label.setMinimumHeight(250)  # 增加最小高度
        self.multi_stage1_label.setCursor(Qt.PointingHandCursor)  # 设置鼠标指针
        self.multi_stage1_label.mouseDoubleClickEvent = lambda event: self.show_full_image(event, self.multi_stage1_label)
        
        self.multi_stage2_label = QLabel("阶段2: 短线补充结果将显示在这里")
        self.multi_stage2_label.setAlignment(Qt.AlignCenter)
        self.multi_stage2_label.setStyleSheet("border: 1px solid #ccc;")
        self.multi_stage2_label.setMinimumHeight(250)  # 增加最小高度
        self.multi_stage2_label.setCursor(Qt.PointingHandCursor)  # 设置鼠标指针
        self.multi_stage2_label.mouseDoubleClickEvent = lambda event: self.show_full_image(event, self.multi_stage2_label)
        
        self.multi_final_label = QLabel("最终结果: 合并结果将显示在这里")
        self.multi_final_label.setAlignment(Qt.AlignCenter)
        self.multi_final_label.setStyleSheet("border: 1px solid #ccc;")
        self.multi_final_label.setMinimumHeight(250)  # 增加最小高度
        self.multi_final_label.setCursor(Qt.PointingHandCursor)  # 设置鼠标指针
        self.multi_final_label.mouseDoubleClickEvent = lambda event: self.show_full_image(event, self.multi_final_label)
        
        self.multi_images_layout.addWidget(self.multi_stage1_label)
        self.multi_images_layout.addWidget(self.multi_stage2_label)
        self.multi_images_layout.addWidget(self.multi_final_label)
        
        # 分步检测统计信息
        self.multi_text_edit = QTextEdit()
        self.multi_text_edit.setReadOnly(True)
        self.multi_text_edit.setMinimumHeight(100)
        
        self.multi_layout.addLayout(self.multi_images_layout)
        self.multi_layout.addWidget(self.multi_text_edit)
        self.tab_widget.addTab(self.multi_tab, "分步检测")
        
        # 日志标签页
        self.log_tab = QWidget()
        self.log_layout = QVBoxLayout(self.log_tab)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_layout.addWidget(self.log_text_edit)
        self.tab_widget.addTab(self.log_tab, "操作日志")
        
        # 添加所有部件到主布局
        main_layout.addWidget(file_group)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(button_group)
        main_layout.addWidget(self.tab_widget)
        
        # 调整比例
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 1)
        main_layout.setStretch(2, 1)
        main_layout.setStretch(3, 10)
        
        # 记录日志
        self.log("应用程序启动成功")

    def browse_file(self):
        """
        浏览并选择PCD文件
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择PCD文件", "", "PCD Files (*.pcd);;All Files (*)"
        )
        
        if file_path:
            self.selected_file = file_path
            self.file_label.setText(f"已选择: {os.path.basename(file_path)}")
            # 启用功能按钮
            self.preview_btn.setEnabled(True)
            self.once_detect_btn.setEnabled(True)
            self.multi_detect_btn.setEnabled(True)
            self.log(f"选择文件: {file_path}")

    def call_api(self, api_type):
        """
        调用API接口
        """
        if not self.selected_file:
            QMessageBox.warning(self, "警告", "请先选择PCD文件")
            return
        
        # 禁用按钮
        self.preview_btn.setEnabled(False)
        self.once_detect_btn.setEnabled(False)
        self.multi_detect_btn.setEnabled(False)
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 获取API URL
        api_url = self.api_endpoints.get(api_type)
        if not api_url:
            QMessageBox.critical(self, "错误", "无效的API类型")
            self._reset_ui_state()
            return
        
        self.log(f"开始调用{self._get_api_name(api_type)} API...")
        
        # 创建并启动线程
        self.thread = APICallThread(api_url, self.selected_file, api_type)
        self.thread.finished.connect(self.handle_api_result)
        self.thread.error.connect(self.handle_api_error)
        self.thread.progress.connect(self.update_progress)
        self.thread.start()

    def handle_api_result(self, result, api_type):
        """
        处理API返回结果
        """
        self.log(f"{self._get_api_name(api_type)} API调用成功")
        
        # 检查结果是否包含message字段
        if "message" not in result or result["message"] != "ok":
            self.handle_api_error(f"API返回错误: {result}")
            return
        
        try:
            if api_type == "preview":
                self._handle_preview_result(result)
            elif api_type == "once":
                self._handle_once_result(result)
            elif api_type == "multi":
                self._handle_multi_result(result)
        except Exception as e:
            self.handle_api_error(f"处理结果时发生错误: {str(e)}")
        finally:
            self._reset_ui_state()

    def _handle_preview_result(self, result):
        """
        处理预览结果
        """
        if "preview_png" in result:
            image_path = result["preview_png"]
            self._display_image(image_path, self.preview_image_label)
            self.log(f"预览图像已显示: {image_path}")
            self.tab_widget.setCurrentWidget(self.preview_tab)

    def _handle_once_result(self, result):
        """
        处理一次检测结果
        """
        if "once_png" in result:
            image_path = result["once_png"]
            self._display_image(image_path, self.once_image_label)
            self.log(f"一次检测图像已显示: {image_path}")
        
        if "line_counts" in result:
            counts_info = "线段统计信息:\n"
            for key, value in result["line_counts"].items():
                counts_info += f"{key}: {value}\n"
            self.once_text_edit.setText(counts_info)
            self.log("一次检测统计信息已显示")
        
        self.tab_widget.setCurrentWidget(self.once_tab)

    def _handle_multi_result(self, result):
        """
        处理分步检测结果
        """
        if "multi_pngs" in result:
            pngs = result["multi_pngs"]
            if "stage1_long" in pngs:
                self._display_image(pngs["stage1_long"], self.multi_stage1_label)
                self.log(f"阶段1图像已显示: {pngs['stage1_long']}")
            
            if "stage2_short" in pngs:
                self._display_image(pngs["stage2_short"], self.multi_stage2_label)
                self.log(f"阶段2图像已显示: {pngs['stage2_short']}")
            
            if "final_merge" in pngs:
                self._display_image(pngs["final_merge"], self.multi_final_label)
                self.log(f"最终结果图像已显示: {pngs['final_merge']}")
        
        if "line_counts" in result:
            counts_info = "线段统计信息:\n"
            for key, value in result["line_counts"].items():
                counts_info += f"{key}: {value}\n"
            self.multi_text_edit.setText(counts_info)
            self.log("分步检测统计信息已显示")
        
        self.tab_widget.setCurrentWidget(self.multi_tab)

    def _display_image(self, image_path, label):
        """
        在标签中显示图像
        """
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            # 缩放图像以适应标签，允许稍大一些以确保清晰度
            # 获取标签的可用大小，减去一些边距
            available_width = label.width() - 20
            available_height = label.height() - 20
            
            # 计算缩放比例，允许图像稍微超出标签以获得更好的清晰度
            scaled_pixmap = pixmap.scaled(
                available_width, available_height, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
            # 保存原始图像路径，用于双击放大
            label.image_path = image_path
        else:
            label.setText(f"图像文件不存在: {image_path}")
            self.log(f"警告: 无法找到图像文件 {image_path}")
            label.image_path = None

    def handle_api_error(self, error_msg):
        """
        处理API调用错误
        """
        self.log(f"错误: {error_msg}")
        QMessageBox.critical(self, "错误", error_msg)
        self._reset_ui_state()

    def update_progress(self, value):
        """
        更新进度条
        """
        self.progress_bar.setValue(value)

    def log(self, message):
        """
        记录日志
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text_edit.append(f"[{timestamp}] {message}")
        # 滚动到底部
        self.log_text_edit.verticalScrollBar().setValue(
            self.log_text_edit.verticalScrollBar().maximum()
        )

    def _get_api_name(self, api_type):
        """
        获取API类型的中文名称
        """
        names = {
            "preview": "点云预览",
            "once": "一次检测",
            "multi": "分步检测"
        }
        return names.get(api_type, api_type)

    def _reset_ui_state(self):
        """
        重置UI状态
        """
        self.preview_btn.setEnabled(True)
        self.once_detect_btn.setEnabled(True)
        self.multi_detect_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def resizeEvent(self, event):
        """
        窗口大小改变时重新调整图像大小
        """
        super().resizeEvent(event)
        # 更新所有显示的图像
        if hasattr(self, 'preview_image_label') and self.preview_image_label.pixmap():
            self._resize_label_image(self.preview_image_label)
        if hasattr(self, 'once_image_label') and self.once_image_label.pixmap():
            self._resize_label_image(self.once_image_label)
        if hasattr(self, 'multi_stage1_label') and self.multi_stage1_label.pixmap():
            self._resize_label_image(self.multi_stage1_label)
        if hasattr(self, 'multi_stage2_label') and self.multi_stage2_label.pixmap():
            self._resize_label_image(self.multi_stage2_label)
        if hasattr(self, 'multi_final_label') and self.multi_final_label.pixmap():
            self._resize_label_image(self.multi_final_label)

    def _resize_label_image(self, label):
        """
        调整标签中的图像大小
        """
        pixmap = label.pixmap()
        if pixmap:
            # 获取标签的可用大小，减去一些边距
            available_width = label.width() - 20
            available_height = label.height() - 20
            
            scaled_pixmap = pixmap.scaled(
                available_width, available_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
            
    def show_full_image(self, event, label):
        """
        双击显示完整图像，优化显示效果，确保图片居中且大小合适
        """
        if hasattr(label, 'image_path') and label.image_path:
            # 创建新窗口显示完整图像
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QScrollArea, QLabel as QDialogLabel, QHBoxLayout, QPushButton, QComboBox
            from PyQt5.QtCore import Qt, QPoint
            
            # 创建对话框
            dialog = QDialog(self)
            dialog.setWindowTitle("图像查看器 - " + os.path.basename(label.image_path))
            dialog.setWindowFlags(Qt.Window | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
            
            # 获取屏幕大小并设置更大的窗口尺寸
            screen = QApplication.primaryScreen().geometry()
            dialog.resize(int(screen.width() * 0.9), int(screen.height() * 0.9))  # 窗口大小为屏幕的90%，转换为整数
            
            # 创建主布局
            main_layout = QVBoxLayout(dialog)
            main_layout.setContentsMargins(5, 5, 5, 5)  # 减少边距
            
            # 创建工具栏
            toolbar_layout = QHBoxLayout()
            
            # 缩放按钮
            zoom_out_btn = QPushButton("缩小")
            zoom_in_btn = QPushButton("放大")
            fit_window_btn = QPushButton("适应窗口")
            original_size_btn = QPushButton("原始大小")
            
            # 缩放比例选择框
            zoom_combo = QComboBox()
            zoom_combo.addItems(["25%", "50%", "75%", "100%", "150%", "200%", "300%", "500%"])
            zoom_combo.setCurrentText("适应窗口")  # 默认选择适应窗口
            
            # 添加到工具栏
            toolbar_layout.addWidget(zoom_out_btn)
            toolbar_layout.addWidget(zoom_in_btn)
            toolbar_layout.addWidget(fit_window_btn)
            toolbar_layout.addWidget(original_size_btn)
            toolbar_layout.addWidget(zoom_combo)
            toolbar_layout.addStretch()
            
            # 创建滚动区域
            scroll_area = QScrollArea()
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # 始终显示水平滚动条
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # 始终显示垂直滚动条
            scroll_area.setWidgetResizable(True)  # 允许小部件调整大小
            
            # 创建容器小部件和布局，确保图像居中
            container_widget = QWidget()
            container_widget.setLayout(QVBoxLayout())
            container_widget.layout().setAlignment(Qt.AlignCenter)
            container_widget.layout().setContentsMargins(0, 0, 0, 0)
            
            # 创建标签显示图像
            full_image_label = QDialogLabel()
            full_image_label.setAlignment(Qt.AlignCenter)  # 图像内容居中
            full_image_label.setScaledContents(True)  # 允许图像缩放以适应标签
            container_widget.layout().addWidget(full_image_label)
            
            # 存储原始图像和缩放信息
            original_pixmap = QPixmap(label.image_path)
            full_image_label.original_pixmap = original_pixmap
            full_image_label.current_scale = 1.0  # 初始缩放比例为100%
            
            # 支持拖动功能
            container_widget.setMouseTracking(True)
            container_widget.dragging = False
            container_widget.last_pos = QPoint()
            
            def mousePressEvent(event):
                if event.button() == Qt.LeftButton and full_image_label.pixmap():
                    container_widget.dragging = True
                    container_widget.last_pos = event.globalPos()
                    container_widget.setCursor(Qt.ClosedHandCursor)
                    event.accept()
                    
            def mouseMoveEvent(event):
                if container_widget.dragging and full_image_label.pixmap():
                    delta = event.globalPos() - container_widget.last_pos
                    h_bar = scroll_area.horizontalScrollBar()
                    v_bar = scroll_area.verticalScrollBar()
                    h_bar.setValue(h_bar.value() - delta.x())
                    v_bar.setValue(v_bar.value() - delta.y())
                    container_widget.last_pos = event.globalPos()
                    event.accept()
                
            def mouseReleaseEvent(event):
                if event.button() == Qt.LeftButton:
                    container_widget.dragging = False
                    container_widget.setCursor(Qt.OpenHandCursor)
                    event.accept()
                    
            # 鼠标滚轮事件处理 - 实现滚轮缩放
            def wheelEvent(event):
                if full_image_label.pixmap():
                    # 获取滚轮事件的角度增量
                    angle_delta = event.angleDelta().y()
                    
                    # 调整缩放比例 (向上滚动放大，向下滚动缩小)
                    if angle_delta > 0:
                        # 向上滚动 - 放大
                        new_scale = full_image_label.current_scale * 1.1
                    else:
                        # 向下滚动 - 缩小
                        new_scale = full_image_label.current_scale * 0.9
                    
                    # 限制缩放范围，避免过小或过大
                    new_scale = max(0.1, min(new_scale, 10.0))  # 限制在10%到1000%之间
                    
                    # 保存滚轮位置，以便缩放后保持该位置为中心
                    mouse_pos = event.pos()
                    viewport = scroll_area.viewport()
                    
                    # 计算缩放前的滚动条位置和比例
                    h_bar = scroll_area.horizontalScrollBar()
                    v_bar = scroll_area.verticalScrollBar()
                    h_ratio = h_bar.value() / (h_bar.maximum() - h_bar.minimum() + viewport.width()) if h_bar.maximum() > h_bar.minimum() else 0.5
                    v_ratio = v_bar.value() / (v_bar.maximum() - v_bar.minimum() + viewport.height()) if v_bar.maximum() > v_bar.minimum() else 0.5
                    
                    # 应用新的缩放比例
                    scale_image(new_scale)
                    
                    # 更新缩放比例选择框
                    percentage = int(new_scale * 100)
                    # 查找最接近的预设缩放比例
                    closest_percentage = min([25, 50, 75, 100, 150, 200, 300, 500], key=lambda x: abs(x - percentage))
                    if abs(closest_percentage - percentage) <= 5:  # 如果差异小于5%，则选择预设值
                        zoom_combo.setCurrentText(f"{closest_percentage}%")
                    
                    # 调整滚动条位置，使缩放以鼠标位置为中心
                    QApplication.processEvents()  # 确保UI已更新
                    new_h_bar = scroll_area.horizontalScrollBar()
                    new_v_bar = scroll_area.verticalScrollBar()
                    
                    new_h_max = new_h_bar.maximum() - new_h_bar.minimum() + viewport.width()
                    new_v_max = new_v_bar.maximum() - new_v_bar.minimum() + viewport.height()
                    
                    new_h_pos = int(new_h_max * h_ratio)
                    new_v_pos = int(new_v_max * v_ratio)
                    
                    new_h_bar.setValue(new_h_pos)
                    new_v_bar.setValue(new_v_pos)
                    
                    # 阻止事件继续传播
                    event.accept()
            
            # 设置鼠标事件处理到容器
            container_widget.mousePressEvent = mousePressEvent
            container_widget.mouseMoveEvent = mouseMoveEvent
            container_widget.mouseReleaseEvent = mouseReleaseEvent
            container_widget.wheelEvent = wheelEvent  # 添加滚轮事件处理
            container_widget.setCursor(Qt.OpenHandCursor)
            
            # 确保full_image_label不会拦截鼠标事件
            full_image_label.setMouseTracking(True)
            full_image_label.mousePressEvent = mousePressEvent
            full_image_label.mouseMoveEvent = mouseMoveEvent
            full_image_label.mouseReleaseEvent = mouseReleaseEvent
            full_image_label.wheelEvent = wheelEvent
            
            # 缩放函数
            def scale_image(scale_factor):
                full_image_label.current_scale = scale_factor
                scaled_pixmap = original_pixmap.scaled(
                    int(original_pixmap.width() * scale_factor),
                    int(original_pixmap.height() * scale_factor),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                full_image_label.setPixmap(scaled_pixmap)
                
                # 保持当前滚动位置，不要强制居中，让用户可以通过滚动条查看图像的所有部分
                
                # 更新缩放比例选择框
                percentage = int(scale_factor * 100)
                for i in range(zoom_combo.count()):
                    if zoom_combo.itemText(i) == f"{percentage}%":
                        zoom_combo.setCurrentIndex(i)
                        break
            
            # 适应窗口函数 - 优化版本
            def fit_to_window():
                # 确保先显示图像，以便获取正确的viewport大小
                if not full_image_label.pixmap():
                    full_image_label.setPixmap(original_pixmap)
                    
                # 强制更新UI以获取正确的尺寸
                dialog.repaint()
                
                # 获取滚动区域的可用大小
                viewport_size = scroll_area.viewport().size()
                
                # 如果图像或视口大小无效，使用默认值
                if original_pixmap.width() <= 0 or original_pixmap.height() <= 0 or viewport_size.width() <= 0 or viewport_size.height() <= 0:
                    scale_image(1.0)
                    return
                    
                image_ratio = original_pixmap.width() / original_pixmap.height()
                viewport_ratio = viewport_size.width() / viewport_size.height()
                
                # 计算缩放比例 - 使用更小的边距，使图像更大
                if image_ratio > viewport_ratio:
                    # 图像较宽，按宽度缩放
                    scale_factor = viewport_size.width() / original_pixmap.width()
                else:
                    # 图像较高，按高度缩放
                    scale_factor = viewport_size.height() / original_pixmap.height()
                
                # 应用缩放（留出很少的边距，让图像更大）
                scale_factor = scale_factor * 0.98  # 几乎填满整个视口
                scale_image(scale_factor)
                
                # 更新缩放比例选择框
                zoom_combo.setCurrentText("适应窗口")
            
            # 连接按钮信号
            zoom_out_btn.clicked.connect(lambda: scale_image(full_image_label.current_scale * 0.8))
            zoom_in_btn.clicked.connect(lambda: scale_image(full_image_label.current_scale * 1.25))
            fit_window_btn.clicked.connect(fit_to_window)
            original_size_btn.clicked.connect(lambda: scale_image(1.0))
            
            # 缩放比例选择框变化
            def on_zoom_changed(text):
                if text != "适应窗口":
                    scale_factor = int(text.replace("%", "")) / 100
                    scale_image(scale_factor)
            zoom_combo.currentTextChanged.connect(on_zoom_changed)
            
            # 窗口大小变化时，重新适应窗口
            def resizeEvent(event):
                if zoom_combo.currentText() == "适应窗口":
                    fit_to_window()
                QDialog.resizeEvent(dialog, event)
            dialog.resizeEvent = resizeEvent
            
            # 设置滚动区域的小部件
            scroll_area.setWidget(container_widget)
            scroll_area.setWidgetResizable(True)  # 允许小部件调整大小，确保图片正确显示
            
            # 添加到主布局
            main_layout.addLayout(toolbar_layout)
            main_layout.addWidget(scroll_area)
            main_layout.setStretch(0, 0)
            main_layout.setStretch(1, 1)
            
            # 显示对话框并立即适应窗口
            dialog.show()
            
            # 确保窗口完全显示后再适应窗口，这样能获取正确的尺寸
            QApplication.processEvents()
            fit_to_window()
            
            # 执行对话框
            dialog.exec_()


if __name__ == "__main__":
    # 设置中文字体
    font = QFont("SimHei", 9)
    
    # 创建应用程序
    app = QApplication(sys.argv)
    app.setFont(font)
    
    # 创建主窗口
    window = PCDVisualizer()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())