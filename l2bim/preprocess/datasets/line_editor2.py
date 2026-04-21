#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
线段编辑器：用于编辑通过Hough变换检测到的线段
"""
import sys
import os
import numpy as np
import cv2
import ezdxf

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QSplitter,
    QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage, QPen, QPainter, QColor, QMouseEvent
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal

# 确保导入路径正确
current_file = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.append(root_path)

# 导入项目模块
from preprocess.datasets.pcd2line_test import run_pcd2line_once
from preprocess.datasets.pcd2lines_step import run_pcd2lines_multistage
from preprocess.geometry.lineseg import LineSegment, LineSegments


class LineEditorCanvas(QWidget):
    """线段编辑画布"""
    pointMoved = pyqtSignal(int, int, tuple)  # 发送：线段索引，端点索引，新坐标
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.parent_widget = parent
        self.img = None
        self.img_pixmap = None
        self.linesegs = None
        self.selected_line_idx = -1  # 选中的线段索引
        self.selected_point_idx = -1  # 选中的端点索引 (0: point_a, 1: point_b)
        self.hovered_line_idx = -1  # 鼠标悬停的线段索引
        self.is_dragging = False
        self.drag_start_pos = QPoint()
        self.click_offset = (0, 0)  # 记录鼠标点击位置与端点的偏移量
        self.point_radius = 8  # 增大端点大小
        self.scale_factor = 1.0  # 缩放因子
        # 设置鼠标跟踪，确保即使不按下鼠标也能接收moveEvent
        self.setMouseTracking(True)
    
    def setParent(self, parent):
        """设置父对象"""
        super().setParent(parent)
        self.parent_widget = parent

    def save_linesegs_to_dxf(linesegs, filename):
        """将 LineSegments 导出为 DXF 矢量线段，方便在 CAD 中使用"""
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()

        # 遍历所有线段，按 line.point_a / line.point_b 写入 DXF
        for line in linesegs.linesegments:
            ax, ay = float(line.point_a[0]), float(line.point_a[1])
            bx, by = float(line.point_b[0]), float(line.point_b[1])
            # 这里默认 Z=0，二维平面
            msp.add_line((ax, ay, 0.0), (bx, by, 0.0))

        doc.saveas(filename)

        
    def set_image(self, img):
        """设置背景图像"""
        self.img = img
        self._update_pixmap()
        self.update()
    
    def set_linesegs(self, linesegs):
        """设置线段集"""
        self.linesegs = linesegs
        self.update()
    
    def _update_pixmap(self):
        """将numpy图像转换为QPixmap"""
        if self.img is None:
            return
        
        height, width = self.img.shape[:2]
        bytes_per_line = width
        
        # 如果是灰度图，转换为RGB
        if len(self.img.shape) == 2:
            q_img = QImage(self.img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            # BGR -> RGB
            rgb_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_img.data, width, height, bytes_per_line * 3, QImage.Format_RGB888)
        
        self.img_pixmap = QPixmap.fromImage(q_img)
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景图像
        if self.img_pixmap:
            scaled_pixmap = self.img_pixmap.scaled(
                self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = (self.width() - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)
            
            # 保存偏移量，用于线段绘制
            offset_x, offset_y = x, y
            scale_x = scaled_pixmap.width() / self.img.shape[1]
            scale_y = scaled_pixmap.height() / self.img.shape[0]
            
            # 绘制线段
            if self.linesegs:
                # 首先绘制所有普通线段（非悬停、非选中）
                for i, line in enumerate(self.linesegs.linesegments):
                    if i != self.selected_line_idx and i != self.hovered_line_idx:
                        # 普通状态：红色，较细
                        pen = QPen(QColor(255, 0, 0), 2)
                        painter.setPen(pen)
                        
                        # 正确计算线段端点坐标
                        x1 = int(line.point_a[0] * scale_x + offset_x)
                        y1 = int(line.point_a[1] * scale_y + offset_y)
                        x2 = int(line.point_b[0] * scale_x + offset_x)
                        y2 = int(line.point_b[1] * scale_y + offset_y)
                        
                        painter.drawLine(x1, y1, x2, y2)
                        
                        # 绘制普通线段的端点：灰色
                        painter.setBrush(QColor(100, 100, 100))
                        painter.drawEllipse(x1 - 4, y1 - 4, 8, 8)
                        painter.drawEllipse(x2 - 4, y2 - 4, 8, 8)
                
                # 然后绘制悬停线段（如果有）
                if self.hovered_line_idx != -1 and self.hovered_line_idx != self.selected_line_idx and self.hovered_line_idx < len(self.linesegs.linesegments):
                    line = self.linesegs.linesegments[self.hovered_line_idx]
                    # 悬停状态：亮黄色，更粗更明显
                    pen = QPen(QColor(255, 255, 0), 4)
                    painter.setPen(pen)
                    
                    x1 = int(line.point_a[0] * scale_x + offset_x)
                    y1 = int(line.point_a[1] * scale_y + offset_y)
                    x2 = int(line.point_b[0] * scale_x + offset_x)
                    y2 = int(line.point_b[1] * scale_y + offset_y)
                    
                    painter.drawLine(x1, y1, x2, y2)
                    
                    # 绘制悬停线段的端点：白色，更大
                    painter.setBrush(QColor(255, 255, 255))
                    painter.drawEllipse(x1 - 6, y1 - 6, 12, 12)
                    painter.drawEllipse(x2 - 6, y2 - 6, 12, 12)
                
                # 最后绘制选中的线段（如果有）
                if self.selected_line_idx != -1 and self.selected_line_idx < len(self.linesegs.linesegments):
                    line = self.linesegs.linesegments[self.selected_line_idx]
                    # 选中状态：亮蓝色，最粗
                    pen = QPen(QColor(0, 150, 255), 4)
                    painter.setPen(pen)
                    
                    x1 = int(line.point_a[0] * scale_x + offset_x)
                    y1 = int(line.point_a[1] * scale_y + offset_y)
                    x2 = int(line.point_b[0] * scale_x + offset_x)
                    y2 = int(line.point_b[1] * scale_y + offset_y)
                    
                    painter.drawLine(x1, y1, x2, y2)
                    
                    # 绘制选中线段的端点，更大更明显
                    # 端点A：如果是选中的端点则为亮绿色，否则为黄色
                    if self.selected_point_idx == 0:
                        painter.setBrush(QColor(0, 255, 0))  # 亮绿色
                    else:
                        painter.setBrush(QColor(255, 255, 0))  # 黄色
                    painter.drawEllipse(x1 - 7, y1 - 7, 14, 14)
                    
                    # 端点B：如果是选中的端点则为亮绿色，否则为黄色
                    if self.selected_point_idx == 1:
                        painter.setBrush(QColor(0, 255, 0))  # 亮绿色
                    else:
                        painter.setBrush(QColor(255, 255, 0))  # 黄色
                    painter.drawEllipse(x2 - 7, y2 - 7, 14, 14)
    
    def mousePressEvent(self, event):
        """鼠标按下事件 - 与悬停检测保持一致，直接在屏幕坐标系中工作"""
        if self.img is None or self.linesegs is None:
            return
        
        # 获取鼠标坐标
        mouse_x = event.x()
        mouse_y = event.y()
        
        # 计算缩放比例和偏移
        if self.img_pixmap:
            # 获取实际显示的图像尺寸和位置
            scaled_pixmap = self.img_pixmap.scaled(
                self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            offset_x = (self.width() - scaled_pixmap.width()) // 2
            offset_y = (self.height() - scaled_pixmap.height()) // 2
            
            # 计算缩放因子
            scale_x = scaled_pixmap.width() / self.img.shape[1]
            scale_y = scaled_pixmap.height() / self.img.shape[0]
            
            # 检查是否点击了某个端点或线段
            clicked = False
            min_dist = float('inf')
            closest_line_idx = -1
            closest_point_idx = -1
            
            # 使用像素容差，直接在屏幕坐标系中工作
            screen_point_tolerance = 15  # 端点容差
            screen_line_tolerance = 10   # 线段容差
            
            # 先检查端点
            for i, line in enumerate(self.linesegs.linesegments):
                # 将线段端点转换为屏幕坐标
                screen_x1 = int(line.point_a[0] * scale_x + offset_x)
                screen_y1 = int(line.point_a[1] * scale_y + offset_y)
                screen_x2 = int(line.point_b[0] * scale_x + offset_x)
                screen_y2 = int(line.point_b[1] * scale_y + offset_y)
                
                # 检查point_a
                dist_a = np.sqrt((screen_x1 - mouse_x) **2 + (screen_y1 - mouse_y)** 2)
                if dist_a < screen_point_tolerance and dist_a < min_dist:
                    min_dist = dist_a
                    closest_line_idx = i
                    closest_point_idx = 0
                    clicked = True
                    # 记录鼠标点击位置与端点的偏移量（屏幕坐标）
                    self.click_offset = (mouse_x - screen_x1, mouse_y - screen_y1)
                
                # 检查point_b
                dist_b = np.sqrt((screen_x2 - mouse_x) **2 + (screen_y2 - mouse_y)** 2)
                if dist_b < screen_point_tolerance and dist_b < min_dist:
                    min_dist = dist_b
                    closest_line_idx = i
                    closest_point_idx = 1
                    clicked = True
                    # 记录鼠标点击位置与端点的偏移量（屏幕坐标）
                    self.click_offset = (mouse_x - screen_x2, mouse_y - screen_y2)
            
            # 如果没有点击端点，检查是否点击了线段
            if not clicked:
                for i, line in enumerate(self.linesegs.linesegments):
                    # 将线段端点转换为屏幕坐标
                    screen_x1 = int(line.point_a[0] * scale_x + offset_x)
                    screen_y1 = int(line.point_a[1] * scale_y + offset_y)
                    screen_x2 = int(line.point_b[0] * scale_x + offset_x)
                    screen_y2 = int(line.point_b[1] * scale_y + offset_y)
                    
                    # 计算点到线段的距离
                    dist = self._point_to_line_distance(mouse_x, mouse_y, 
                                                       screen_x1, screen_y1,
                                                       screen_x2, screen_y2)
                    if dist < screen_line_tolerance and dist < min_dist:
                        min_dist = dist
                        closest_line_idx = i
                        closest_point_idx = -1  # 选中整个线段
                        clicked = True
                # 重置偏移量
                self.click_offset = (0, 0)
            
            # 更新选中状态
            if clicked:
                self.selected_line_idx = closest_line_idx
                self.selected_point_idx = closest_point_idx
                # 开始拖动
                if self.selected_point_idx != -1:
                    self.is_dragging = True
                    self.drag_start_pos = event.pos()
                # 更新属性面板
                if hasattr(self.parent_widget, 'update_selected_line_info'):
                    self.parent_widget.update_selected_line_info()
            else:
                # 如果都没有点击，取消选择
                self.selected_line_idx = -1
                self.selected_point_idx = -1
                self.click_offset = (0, 0)  # 重置偏移量
                if hasattr(self.parent_widget, 'clear_selected_line_info'):
                    self.parent_widget.clear_selected_line_info()
            
            self.update()
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件 - 使用偏移量实现平滑拖拽"""
        # 处理拖动情况
        if self.is_dragging and self.selected_line_idx != -1 and self.selected_point_idx != -1:
            # 计算缩放比例
            if self.img_pixmap:
                # 获取实际显示的图像尺寸和位置
                scaled_pixmap = self.img_pixmap.scaled(
                    self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                offset_x = (self.width() - scaled_pixmap.width()) // 2
                offset_y = (self.height() - scaled_pixmap.height()) // 2
                
                scale_x = scaled_pixmap.width() / self.img.shape[1]
                scale_y = scaled_pixmap.height() / self.img.shape[0]
                
                # 使用偏移量计算新的端点屏幕坐标
                # 鼠标当前位置减去偏移量，得到端点应该在的屏幕位置
                screen_endpoint_x = event.x() - self.click_offset[0]
                screen_endpoint_y = event.y() - self.click_offset[1]
                
                # 将屏幕坐标转换为图像坐标
                img_x = (screen_endpoint_x - offset_x) / scale_x
                img_y = (screen_endpoint_y - offset_y) / scale_y
                
                # 确保索引有效
                if self.selected_line_idx >= 0 and self.selected_line_idx < len(self.linesegs.linesegments):
                    # 更新线段端点
                    line = self.linesegs.linesegments[self.selected_line_idx]
                    if self.selected_point_idx == 0:
                        line.point_a = np.array([img_x, img_y])
                    else:
                        line.point_b = np.array([img_x, img_y])
                    
                    # 更新线段的方向和长度
                    line.direction = line.point_b - line.point_a
                    line.direction = line.direction.astype(float)
                    if np.linalg.norm(line.direction) > 0:
                        line.direction /= np.linalg.norm(line.direction)
                    line.length = line.get_length()
                    
                    # 发送信号
                    self.pointMoved.emit(self.selected_line_idx, self.selected_point_idx, (img_x, img_y))
                    
                    # 更新属性面板
                    if hasattr(self.parent_widget, 'update_line_properties'):
                        self.parent_widget.update_line_properties(line)
                    
                    self.update()  # 重绘界面以显示更新后的线段
        else:
            # 处理悬停情况：更新悬停的线段
            self._update_hovered_line(event)
            
    def _update_hovered_line(self, event):
        """更新鼠标悬停的线段 - 完全重写坐标转换和检测逻辑"""
        if self.img is None or self.linesegs is None:
            if self.hovered_line_idx != -1:
                self.hovered_line_idx = -1
                self.update()
            return
        
        # 计算缩放比例和偏移 - 简化计算确保准确性
        if not self.img_pixmap:
            return
            
        # 获取实际显示的图像尺寸和位置
        scaled_pixmap = self.img_pixmap.scaled(
            self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        offset_x = (self.width() - scaled_pixmap.width()) // 2
        offset_y = (self.height() - scaled_pixmap.height()) // 2
        
        # 计算缩放因子
        scale_x = scaled_pixmap.width() / self.img.shape[1]
        scale_y = scaled_pixmap.height() / self.img.shape[0]
        
        # 关键修改：直接在屏幕坐标系统中计算距离，避免坐标转换错误
        mouse_x = event.x()
        mouse_y = event.y()
        
        # 查找最近的线段
        min_dist = float('inf')
        closest_line_idx = -1
        # 使用像素容差，直接在屏幕坐标系中工作
        screen_tolerance = 12  # 调整为合适的像素值
        
        for i, line in enumerate(self.linesegs.linesegments):
            # 跳过已选中的线段
            if i == self.selected_line_idx:
                continue
                
            # 将线段端点转换为屏幕坐标
            screen_x1 = int(line.point_a[0] * scale_x + offset_x)
            screen_y1 = int(line.point_a[1] * scale_y + offset_y)
            screen_x2 = int(line.point_b[0] * scale_x + offset_x)
            screen_y2 = int(line.point_b[1] * scale_y + offset_y)
            
            # 直接在屏幕坐标系中计算点到线段的距离
            dist = self._point_to_line_distance(mouse_x, mouse_y, 
                                              screen_x1, screen_y1,
                                              screen_x2, screen_y2)
            
            # 同时检查是否靠近端点
            dist_a = np.sqrt((screen_x1 - mouse_x) **2 + (screen_y1 - mouse_y)** 2)
            dist_b = np.sqrt((screen_x2 - mouse_x) **2 + (screen_y2 - mouse_y)** 2)
            
            # 取距离线段和端点的最小距离
            min_segment_dist = min(dist, dist_a, dist_b)
            
            if min_segment_dist < screen_tolerance and min_segment_dist < min_dist:
                min_dist = min_segment_dist
                closest_line_idx = i
        
        # 更新悬停状态
        if self.hovered_line_idx != closest_line_idx:
            self.hovered_line_idx = closest_line_idx
            self.update()  # 重绘以显示高亮效果
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        self.is_dragging = False
        # 重置偏移量
        self.click_offset = (0, 0)
        # 释放鼠标后检查悬停状态
        self._update_hovered_line(event)
    
    def _point_to_line_distance(self, x, y, x1, y1, x2, y2):
        """计算点到线段的最短距离 - 优化版本，修复坐标计算问题"""
        # 计算线段向量和点到起点的向量
        line_vec_x = x2 - x1
        line_vec_y = y2 - y1
        point_vec_x = x - x1
        point_vec_y = y - y1
        
        # 检查方向向量是否为零向量
        if line_vec_x == 0 and line_vec_y == 0:
            # 线段退化为点，返回点到点的距离
            return np.sqrt(point_vec_x*point_vec_x + point_vec_y*point_vec_y)
        
        # 计算点积
        dot_product = point_vec_x * line_vec_x + point_vec_y * line_vec_y
        # 线段长度的平方
        line_len_sq = line_vec_x * line_vec_x + line_vec_y * line_vec_y
        
        # 计算投影参数，但避免除以零
        t = max(0, min(1, dot_product / line_len_sq)) if line_len_sq != 0 else 0
        
        # 计算投影点坐标
        projection_x = x1 + t * line_vec_x
        projection_y = y1 + t * line_vec_y
        
        # 计算点到投影点的距离
        dx = x - projection_x
        dy = y - projection_y
        return np.sqrt(dx*dx + dy*dy)
    
    def get_linesegs(self):
        """获取当前线段集"""
        return self.linesegs


class LineEditorApp(QMainWindow):
    """线段编辑器主应用"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.result_data = None  # 存储处理结果
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("线段编辑器")
        self.setGeometry(100, 100, 1200, 800)
        
        # 主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)
        
        # 顶部工具栏
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)
        
        # 加载点云按钮
        self.load_pcd_btn = QPushButton("加载点云")
        self.load_pcd_btn.clicked.connect(self.load_pcd)
        toolbar_layout.addWidget(self.load_pcd_btn)
        
        # 加载配置按钮
        self.load_cfg_btn = QPushButton("加载配置")
        self.load_cfg_btn.clicked.connect(self.load_cfg)
        toolbar_layout.addWidget(self.load_cfg_btn)
        
        # 检测方法选择
        self.method_combo = QComboBox()
        self.method_combo.addItems(["单次检测", "多阶段检测"])
        toolbar_layout.addWidget(QLabel("检测方法:"))
        toolbar_layout.addWidget(self.method_combo)
        
        # 运行检测按钮
        self.run_detection_btn = QPushButton("运行检测")
        self.run_detection_btn.clicked.connect(self.run_detection)
        toolbar_layout.addWidget(self.run_detection_btn)
        
        # 保存线段按钮
        self.save_lines_btn = QPushButton("保存线段")
        self.save_lines_btn.clicked.connect(self.save_lines)
        toolbar_layout.addWidget(self.save_lines_btn)
        
        toolbar_layout.addStretch()
        main_layout.addWidget(toolbar_widget)
        
        # 中间分隔器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧画布
        self.canvas = LineEditorCanvas(self)  # 在创建时直接传入父对象
        self.canvas.pointMoved.connect(self.on_point_moved)
        splitter.addWidget(self.canvas)
        
        # 右侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # 线段属性组
        self.line_properties_group = QGroupBox("线段属性")
        line_props_layout = QGridLayout(self.line_properties_group)
        
        # 选中线段信息
        self.selected_line_info = QLabel("未选中线段")
        line_props_layout.addWidget(self.selected_line_info, 0, 0, 1, 2)
        
        # 端点A坐标
        line_props_layout.addWidget(QLabel("端点A X:"), 1, 0)
        self.point_a_x = QDoubleSpinBox()
        self.point_a_x.setRange(0, 10000)
        self.point_a_x.setDecimals(2)
        self.point_a_x.valueChanged.connect(self.on_point_a_x_changed)
        line_props_layout.addWidget(self.point_a_x, 1, 1)
        
        line_props_layout.addWidget(QLabel("端点A Y:"), 2, 0)
        self.point_a_y = QDoubleSpinBox()
        self.point_a_y.setRange(0, 10000)
        self.point_a_y.setDecimals(2)
        self.point_a_y.valueChanged.connect(self.on_point_a_y_changed)
        line_props_layout.addWidget(self.point_a_y, 2, 1)
        
        # 端点B坐标
        line_props_layout.addWidget(QLabel("端点B X:"), 3, 0)
        self.point_b_x = QDoubleSpinBox()
        self.point_b_x.setRange(0, 10000)
        self.point_b_x.setDecimals(2)
        self.point_b_x.valueChanged.connect(self.on_point_b_x_changed)
        line_props_layout.addWidget(self.point_b_x, 3, 1)
        
        line_props_layout.addWidget(QLabel("端点B Y:"), 4, 0)
        self.point_b_y = QDoubleSpinBox()
        self.point_b_y.setRange(0, 10000)
        self.point_b_y.setDecimals(2)
        self.point_b_y.valueChanged.connect(self.on_point_b_y_changed)
        line_props_layout.addWidget(self.point_b_y, 4, 1)
        
        # 线段长度
        line_props_layout.addWidget(QLabel("长度:"), 5, 0)
        self.line_length = QLabel("0.00")
        line_props_layout.addWidget(self.line_length, 5, 1)
        
        # 操作按钮
        self.delete_line_btn = QPushButton("删除线段")
        self.delete_line_btn.clicked.connect(self.delete_line)
        line_props_layout.addWidget(self.delete_line_btn, 6, 0, 1, 2)
        
        control_layout.addWidget(self.line_properties_group)
        control_layout.addStretch()
        
        splitter.addWidget(control_panel)
        splitter.setSizes([800, 400])  # 设置初始大小
        
        main_layout.addWidget(splitter)
        
        # 状态变量
        self.pcd_path = ""
        self.cfg_path = ""
    
    def load_pcd(self):
        """加载点云文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择点云文件", "", "点云文件 (*.pcd *.ply)")
        if file_path:
            self.pcd_path = file_path
            self.statusBar().showMessage(f"已加载点云: {os.path.basename(file_path)}")
    
    def load_cfg(self):
        """加载配置文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择配置文件", "", "配置文件 (*.yaml *.yml)")
        if file_path:
            self.cfg_path = file_path
            self.statusBar().showMessage(f"已加载配置: {os.path.basename(file_path)}")
    
    def run_detection(self):
        """运行线段检测"""
        if not self.pcd_path or not self.cfg_path:
            QMessageBox.warning(self, "警告", "请先加载点云和配置文件")
            return
        
        try:
            self.statusBar().showMessage("正在运行线段检测...")
            
            # 禁用按钮防止重复点击
            self.run_detection_btn.setEnabled(False)
            
            # 清除之前的选中状态和结果
            self.canvas.selected_line_idx = -1
            self.canvas.selected_point_idx = -1
            self.canvas.hovered_line_idx = -1  # 清除悬停状态
            self.canvas.is_dragging = False
            self.clear_selected_line_info()
            
            # 强制清空之前的数据
            self.result_data = None
            
            # 根据选择的方法运行检测
            if self.method_combo.currentText() == "单次检测":
                self.result_data = run_pcd2line_once(
                    self.pcd_path, self.cfg_path, show=False
                )
                linesegs = self.result_data["linesegs"]
            else:  # 多阶段检测
                self.result_data = run_pcd2lines_multistage(
                    self.pcd_path, self.cfg_path, show=False
                )
                linesegs = self.result_data["lines_all"]
            
            # 显示结果 - 先设置线段再设置图像，确保绘制正确
            self.canvas.set_linesegs(linesegs)
            self.canvas.set_image(self.result_data["img"])
            
            # 强制更新画布
            self.canvas.update()
            
            self.statusBar().showMessage(f"检测完成，共发现 {len(linesegs.linesegments)} 条线段")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"检测过程中出现错误: {str(e)}")
            self.statusBar().showMessage("检测失败")
        finally:
            # 无论成功失败，重新启用按钮
            self.run_detection_btn.setEnabled(True)
    
    def on_point_moved(self, line_idx, point_idx, new_pos):
        """当线段端点被移动时更新属性面板"""
        if self.canvas.linesegs:
            line = self.canvas.linesegs.linesegments[line_idx]
            self.update_line_properties(line)
    
    def update_line_properties(self, line):
        """更新线段属性面板"""
        # 禁用信号以避免循环更新
        self.point_a_x.blockSignals(True)
        self.point_a_y.blockSignals(True)
        self.point_b_x.blockSignals(True)
        self.point_b_y.blockSignals(True)
        
        self.point_a_x.setValue(line.point_a[0])
        self.point_a_y.setValue(line.point_a[1])
        self.point_b_x.setValue(line.point_b[0])
        self.point_b_y.setValue(line.point_b[1])
        self.line_length.setText(f"{line.get_length():.2f}")
        
        # 重新启用信号
        self.point_a_x.blockSignals(False)
        self.point_a_y.blockSignals(False)
        self.point_b_x.blockSignals(False)
        self.point_b_y.blockSignals(False)
    
    def update_selected_line_info(self):
        """更新选中线段信息"""
        if self.canvas.selected_line_idx != -1 and self.canvas.linesegs:
            line_idx = self.canvas.selected_line_idx
            if line_idx < len(self.canvas.linesegs.linesegments):
                line = self.canvas.linesegs.linesegments[line_idx]
                self.selected_line_info.setText(f"选中线段 #{line_idx + 1}")
                self.update_line_properties(line)
    
    def clear_selected_line_info(self):
        """清除选中线段信息"""
        self.selected_line_info.setText("未选中线段")
        # 禁用信号以避免循环更新
        self.point_a_x.blockSignals(True)
        self.point_a_y.blockSignals(True)
        self.point_b_x.blockSignals(True)
        self.point_b_y.blockSignals(True)
        
        self.point_a_x.setValue(0)
        self.point_a_y.setValue(0)
        self.point_b_x.setValue(0)
        self.point_b_y.setValue(0)
        self.line_length.setText("0.00")
        
        # 重新启用信号
        self.point_a_x.blockSignals(False)
        self.point_a_y.blockSignals(False)
        self.point_b_x.blockSignals(False)
        self.point_b_y.blockSignals(False)
    
    def on_point_a_x_changed(self, value):
        """端点A X坐标改变"""
        if self.canvas.selected_line_idx != -1:
            line = self.canvas.linesegs.linesegments[self.canvas.selected_line_idx]
            line.point_a[0] = value
            line.direction = line.point_b - line.point_a
            line.direction /= np.linalg.norm(line.direction)
            line.length = line.get_length()
            self.line_length.setText(f"{line.get_length():.2f}")
            self.canvas.update()
    
    def on_point_a_y_changed(self, value):
        """端点A Y坐标改变"""
        if self.canvas.selected_line_idx != -1:
            line = self.canvas.linesegs.linesegments[self.canvas.selected_line_idx]
            line.point_a[1] = value
            line.direction = line.point_b - line.point_a
            line.direction /= np.linalg.norm(line.direction)
            line.length = line.get_length()
            self.line_length.setText(f"{line.get_length():.2f}")
            self.canvas.update()
    
    def on_point_b_x_changed(self, value):
        """端点B X坐标改变"""
        if self.canvas.selected_line_idx != -1:
            line = self.canvas.linesegs.linesegments[self.canvas.selected_line_idx]
            line.point_b[0] = value
            line.direction = line.point_b - line.point_a
            line.direction /= np.linalg.norm(line.direction)
            line.length = line.get_length()
            self.line_length.setText(f"{line.get_length():.2f}")
            self.canvas.update()
    
    def on_point_b_y_changed(self, value):
        """端点B Y坐标改变"""
        if self.canvas.selected_line_idx != -1:
            line = self.canvas.linesegs.linesegments[self.canvas.selected_line_idx]
            line.point_b[1] = value
            line.direction = line.point_b - line.point_a
            line.direction /= np.linalg.norm(line.direction)
            line.length = line.get_length()
            self.line_length.setText(f"{line.get_length():.2f}")
            self.canvas.update()
    
    def delete_line(self):
        """删除选中的线段"""
        if self.canvas.selected_line_idx != -1:
            reply = QMessageBox.question(
                self, "确认", "确定要删除这条线段吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                # 删除线段
                del self.canvas.linesegs.linesegments[self.canvas.selected_line_idx]
                self.canvas.selected_line_idx = -1
                self.canvas.selected_point_idx = -1
                self.selected_line_info.setText("未选中线段")
                self.canvas.update()
                self.statusBar().showMessage(f"线段已删除，剩余 {len(self.canvas.linesegs.linesegments)} 条线段")
    
    def save_lines(self):
        """保存编辑后的线段（支持 .pkl 和 .dxf）"""
        if not self.canvas.linesegs:
            QMessageBox.warning(self, "警告", "没有可保存的线段")
            return

        # 允许用户选择保存为 pkl 或 dxf
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存线段",
            "",
            "线段数据文件 (*.pkl);;DXF 文件 (*.dxf)"
        )

        if not file_path:
            return

        try:
            # 根据后缀决定保存格式
            lower_path = file_path.lower()
            if lower_path.endswith(".dxf"):
                # 导出为 DXF（CAD 矢量线段）
                save_linesegs_to_dxf(self.canvas.linesegs, file_path)
                self.statusBar().showMessage(f"DXF 线段已保存到: {file_path}")
                QMessageBox.information(self, "成功", "DXF 矢量线段导出成功")
            else:
                # 默认按原来的逻辑，保存为 pkl
                if not lower_path.endswith(".pkl"):
                    file_path = file_path + ".pkl"
                self.canvas.linesegs.save_to_file(file_path)
                self.statusBar().showMessage(f"线段数据已保存到: {file_path}")
                QMessageBox.information(self, "成功", "线段数据(pkl)保存成功")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LineEditorApp()
    window.show()
    sys.exit(app.exec_())