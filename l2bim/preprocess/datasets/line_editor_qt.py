#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
线段编辑器PyQt5组件：封装line_editor3功能，用于线段的可视化和编辑
"""
import os
import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QDoubleSpinBox, QFileDialog, QFrame,
    QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage, QPen, QPainter, QColor
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
current_file = os.path.abspath(__file__)
# 向上跳三级：datasets → preprocess → 根目录
root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
#print("修正后root_path:", root_path)  # 验证是否为项目根目录
sys.path.append(root_path)
from preprocess.datasets.pcd2line_test import run_pcd2line_once   

# 确保导入路径正确
current_file = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.append(root_path)

# 导入项目模块
from preprocess.geometry.lineseg import LineSegment, LineSegments
import ezdxf


class LineItemWidget(QWidget):
    """线段列表项组件"""
    
    def __init__(self, line_id, line_length, parent=None):
        super().__init__(parent)
        self.line_id = line_id
        self.line_length = line_length
        self.is_selected = False
        
        # 设置布局
        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(5, 5, 5, 5)
        self.layout().setSpacing(10)
        
        # 设置样式
        self.setStyleSheet("""
            QWidget {
                background-color: #333333;
                border: 1px solid #555555;
                border-radius: 3px;
                font-family: 'SimHei', 'Microsoft YaHei', 'sans-serif';
                font-size:14px;
            }
            QWidget:hover {
                background-color: #404040;
                border: 1px solid #666666;
            }
        """)
        
        # 添加ID标签
        self.id_label = QLabel(f"线段 ID: {line_id}")
        self.id_label.setStyleSheet("color: #ffffff; font-weight: bold;font-family: 'SimHei', 'Microsoft YaHei', 'sans-serif';")
        self.id_label.setMinimumWidth(80)
        self.id_label.setMaximumWidth(120)
        self.layout().addWidget(self.id_label)
        
        # 添加弹性空间 - 左侧
        self.layout().addStretch()
        
        # 添加长度标签
        self.length_label = QLabel(f"长度: {line_length:.4f}")
        self.length_label.setStyleSheet("color: #ffffff;font-family: 'SimHei', 'Microsoft YaHei', 'sans-serif';")
        self.layout().addWidget(self.length_label)
        
        # 添加弹性空间 - 右侧
        self.layout().addStretch()
        
        # 设置大小策略，使其能够水平扩展
        self.setSizePolicy(self.sizePolicy().Expanding, self.sizePolicy().Fixed)
    
    def set_selected(self, selected):
        """设置选中状态"""
        self.is_selected = selected
        if selected:
            self.setStyleSheet("""
                QWidget {
                    background-color: #556b2f;
                    border: 2px solid #9acd32;
                    border-radius: 3px;
                    font-family: 'SimHei', 'Microsoft YaHei', 'sans-serif';
                    font-size:14px;
                }
            """)
        else:
            self.setStyleSheet("""
                QWidget {
                    background-color: #333333;
                    border: 1px solid #555555;
                    border-radius: 3px;
                    font-family: 'SimHei', 'Microsoft YaHei', 'sans-serif';
                    font-size:14px;
                }
                QWidget:hover {
                    background-color: #404040;
                    border: 1px solid #666666;
                }
            """)
    
    def mousePressEvent(self, event):
        """鼠标点击事件，发出选中信号"""
        if hasattr(self.parent(), 'parent_widget'):
            self.parent().parent_widget.on_line_item_clicked(self.line_id)
        super().mousePressEvent(event)


class LineEditorComponent(QWidget):
    """线段编辑器PyQt5组件
    
    提供以下功能：
    1. 显示二维图像
    2. 编辑线段（移动、创建、删除）
    3. 实时显示和修改线段属性
    
    信号：
    - lineSelected: 当选择一条线段时发出，参数为选中的LineSegment对象
    - lineModified: 当修改一条线段时发出，参数为修改后的LineSegment对象
    - lineDeleted: 当删除一条线段时发出，参数为被删除线段的索引
    """
    # 定义信号
    lineSelected = pyqtSignal(object)  # 参数为选中的LineSegment对象
    lineModified = pyqtSignal(object)  # 参数为修改后的LineSegment对象
    lineDeleted = pyqtSignal(int)      # 参数为被删除线段的索引
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 设置深色主题背景
        self.setStyleSheet("background-color: #2d2d2d; color: #ffffff;")
        
        # 初始化数据
        self.img = None
        self.linesegs = None
        self.line_items = []  # 存储线段列表项组件
        self.selected_line_id = -1  # 当前选中的线段ID
        
        # 初始化UI
        self.init_ui()
        
        # 设置线段列表内容窗口的父组件引用
        self.line_list_widget.parent_widget = self
    
    def update_line_list(self):
        """更新线段列表"""
        # 清空现有线段项
        for item in self.line_items:
            self.line_list_layout.removeWidget(item)
            item.deleteLater()
        self.line_items = []
        
        # 如果没有线段数据，直接返回
        if self.linesegs is None:
            return
        
        # 创建新的线段项
        for i, line in enumerate(self.linesegs.linesegments):
            # 使用线段对象的length属性
            length = line.length
            
            # 创建线段项组件
            line_item = LineItemWidget(i, length)
            
            # 如果是当前选中的线段，设置为选中状态
            if i == self.selected_line_id:
                line_item.set_selected(True)
            
            # 添加到布局和列表
            self.line_list_layout.addWidget(line_item)
            self.line_items.append(line_item)
    
    def on_line_item_clicked(self, line_id):
        """处理线段项点击事件"""
        # 更新选中状态
        for item in self.line_items:
            item.set_selected(item.line_id == line_id)
        
        # 更新当前选中的线段ID
        self.selected_line_id = line_id
        
        # 更新画布上的选中状态
        if self.canvas and 0 <= line_id < len(self.linesegs.linesegments):
            self.canvas.selected_line_idx = line_id
            self.canvas.selected_point_idx = -1
            self.canvas.update()
            
            # 更新属性面板
            if hasattr(self, 'update_selected_line_info'):
                self.update_selected_line_info()
    
    def init_ui(self):
        """初始化UI组件"""
        # 主布局 - 恢复为水平布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 左侧画布容器
        canvas_container = QWidget()
        canvas_container.setStyleSheet("background-color: #1a1a1a;")
        
        # 初始化属性面板组件在for循环中完成，这里不再设置为None
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        # 画布标题
        canvas_title = QLabel("线段编辑")
        canvas_title.setStyleSheet("color: #ffffff; background-color: #1a1a1a; padding: 5px; font-weight: bold;")
        canvas_title.setAlignment(Qt.AlignCenter)
        
        # 创建画布
        self.canvas = LineEditorCanvas(self)
        self.canvas.pointMoved.connect(self.on_point_moved)
        self.canvas.dragFinished.connect(self.on_drag_finished)
        
        # 将标题和画布添加到容器布局
        canvas_layout.addWidget(canvas_title)
        canvas_layout.addWidget(self.canvas)
        
        # 右侧属性面板
        self.prop_group = QGroupBox("线段属性")
        self.prop_group.setStyleSheet("""
            QGroupBox {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 0px;
                margin-top: 0px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                color: #ffffff;
            }
        """)
        prop_layout = QVBoxLayout(self.prop_group)
        #prop_layout.setContentsMargins(10, 10, 10, 10)  # 添加内边距
        
        prop_layout.addSpacing(30)
        # 线段列表滑动框
        self.line_list_scroll_area = QScrollArea()
        self.line_list_scroll_area.setWidgetResizable(True)
        self.line_list_scroll_area.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        
        # 滑动框内容窗口
        self.line_list_widget = QWidget()
        self.line_list_widget.setStyleSheet("background-color: #1a1a1a;")
        self.line_list_layout = QVBoxLayout(self.line_list_widget)
        self.line_list_layout.setAlignment(Qt.AlignTop)
        self.line_list_layout.setContentsMargins(5, 5, 5, 5)
        self.line_list_layout.setSpacing(5)
        
        # 设置滑动框的内容窗口
        self.line_list_scroll_area.setWidget(self.line_list_widget)
        # 设置滑动框可调整内容窗口大小
        self.line_list_scroll_area.setWidgetResizable(True)
        
        # 设置滑动框固定高度为450像素
        self.line_list_scroll_area.setFixedHeight(420)
        
        # 将滑动框添加到属性面板布局
        prop_layout.addWidget(self.line_list_scroll_area)
        
        # 添加分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #555;")
        prop_layout.addWidget(separator)

        prop_layout.addSpacing(10)
        
        # 创建统一的标签样式
        label_style = "font-size: 14px; font-family: 'SimHei', 'Microsoft YaHei', 'sans-serif'; color: #FFFFFF;"
        
        # 创建统一的输入框样式
        spinbox_style = """
            QDoubleSpinBox {
                background-color: #404040;
                border: 1px solid #666;
                color: #ffffff;
                padding: 4px;
                font-size: 12px;
                width: 150px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: #555;
                color: #ffffff;
                width: 20px;
            }
        """
        
        # 使用for循环创建端点坐标输入框，参考lidar_gui.py的实现方式
        coord_params = [
            ("端点A x:", "point_a_x", self.on_point_a_x_changed),
            ("端点A y:", "point_a_y", self.on_point_a_y_changed),
            ("端点B x:", "point_b_x", self.on_point_b_x_changed),
            ("端点B y:", "point_b_y", self.on_point_b_y_changed)
        ]
        
        for label_text, attr_name, callback in coord_params:
            # 创建水平布局来放置标签和输入框
            h_layout = QHBoxLayout()
            
            # 创建标签
            label = QLabel(label_text)
            label.setStyleSheet(label_style)
            label.setMinimumWidth(120)  # 增加标签宽度，确保文字完全显示
            label.setMaximumWidth(120)
            #label.setFixedWidth(80)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # 左对齐
            label.setStyleSheet("font-size: 14px; font-family: 'SimHei', 'Microsoft YaHei', 'sans-serif'; color: #FFFFFF;")
            h_layout.addWidget(label)
            
            # 创建输入框
            spin_box = QDoubleSpinBox()
            spin_box.setDecimals(4)
            spin_box.setMinimum(-10000.0)
            spin_box.setMaximum(100000.0)
            spin_box.setSingleStep(1.0)
            spin_box.setStyleSheet(spinbox_style)
            spin_box.valueChanged.connect(callback)
            h_layout.addWidget(spin_box)
            
            # 将输入框保存为实例属性
            setattr(self, attr_name, spin_box)
            
            # 添加水平布局到垂直布局
            prop_layout.addLayout(h_layout)
            prop_layout.addSpacing(8)  #垂直组件之间的间距
        
        # 线段长度 - 使用水平布局
        length_layout = QHBoxLayout()
        length_label = QLabel("线段长度:")
        length_label.setStyleSheet(label_style)
        length_label.setMinimumWidth(120)  # 增加标签宽度，确保文字完全显示
        length_label.setMaximumWidth(120)
        #length_label.setFixedWidth(80)
        length_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # 左对齐
        length_layout.addWidget(length_label)
        
        self.line_length = QLabel("0.0000")
        self.line_length.setStyleSheet("color: #ffffff; background-color: #404040; border: 1px solid #666; padding: 4px; font-size: 12px;")
        self.line_length.setFixedHeight(24)
        self.line_length.setFixedWidth(180)
        length_layout.addWidget(self.line_length)
        
        prop_layout.addLayout(length_layout)
        
        # 添加间距
        prop_layout.addSpacing(10)
        
        # 删除按钮
        self.delete_btn = QPushButton("删除线段")
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                border: 1px solid #388e3c;
                color: white;
                padding: 8px 16px;
                text-align: center;
                font-size: 14px;
                font-family: 'SimHei', 'Microsoft YaHei', 'sans-serif';
                margin: 4px 2px;
                width: 100%;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:pressed {
                background-color: #bd2130;
            }
        """)
        self.delete_btn.clicked.connect(self.delete_line)
        prop_layout.addWidget(self.delete_btn)
        
        # 添加分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #555;")
        prop_layout.addWidget(separator)

        prop_layout.addSpacing(10)
        
        # 创建水平布局来放置保存按钮
        save_buttons_layout = QHBoxLayout()
        save_buttons_layout.setSpacing(10)  # 设置按钮间距
        
        # 添加保存图片按钮
        self.save_image_button = QPushButton("保存图片")
        self.save_image_button.setStyleSheet("background-color: #007bff; color: white; padding: 5px;")
        self.save_image_button.clicked.connect(self.on_save_image)
        save_buttons_layout.addWidget(self.save_image_button)
        
        # 添加保存DXF按钮
        self.save_dxf_button = QPushButton("保存DXF文件")
        self.save_dxf_button.setStyleSheet("background-color: #28a745; color: white; padding: 5px;")
        self.save_dxf_button.clicked.connect(self.on_save_dxf)
        save_buttons_layout.addWidget(self.save_dxf_button)
        
        # 将水平布局添加到属性布局中
        prop_layout.addLayout(save_buttons_layout)
        
        # 添加拉伸因子
        prop_layout.addStretch(1)
        
        self.prop_group.setLayout(prop_layout)
        
        # 将组件添加到主布局 - 左侧画布，右侧属性面板
        main_layout.addWidget(canvas_container, 3)
        main_layout.addWidget(self.prop_group, 1)
        
        # 设置布局
        self.setLayout(main_layout)
    
    def set_image(self, img):
        """设置背景图像
        
        Args:
            img: numpy格式的二维图像数组，支持灰度图和RGB图像
        """
        # 验证输入是否为numpy数组
        if not isinstance(img, np.ndarray):
            raise TypeError("Input image must be a numpy array")
        
        # 验证图像维度是否正确（灰度图或RGB图）
        if len(img.shape) not in [2, 3]:
            raise ValueError("Input image must be 2D (grayscale) or 3D (RGB/BGR)")
        
        # 存储图像并传递给画布
        self.img = img.copy()  # 创建副本以避免外部修改
        self.canvas.set_image(self.img)
        
        # 直接使用内部存储的图像，不再依赖LineEditModel
    
    def set_linesegs(self, linesegs):
        """设置线段集
        
        Args:
            linesegs: LineSegments对象，包含多个LineSegment
        """
        # 验证输入是否为LineSegments对象
        if not hasattr(linesegs, 'linesegments'):
            raise TypeError("Input must be a LineSegments object with linesegments attribute")
        
        # 存储线段集并传递给画布
        self.linesegs = linesegs
        self.canvas.set_linesegs(linesegs)
        self.clear_selected_line_info()
        
        # 直接使用内部存储的线段集，不再依赖LineEditModel
        
        # 更新线段列表
        self.update_line_list()
    
    def save_image(self, file_path):
        """保存画布内容为图片
        
        Args:
            file_path: 保存的文件路径
        
        Returns:
            bool: 保存是否成功
        """
        return self.canvas.save_image(file_path)
    
    def save_dxf(self, file_path):
        """保存线段为DXF文件
        
        Args:
            file_path: 保存的文件路径
        
        Returns:
            bool: 保存是否成功
        """
        try:
            if self.linesegs is None:
                print("没有线段可保存")
                return False
            
            # 确保目录存在
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # 创建DXF文档
            doc = ezdxf.new("R2010")
            msp = doc.modelspace()
            
            # 获取图像高度以进行坐标翻转（如果有图像的话）
            img_height = 0
            if self.img is not None:
                img_height = self.img.shape[0]
            
            # 添加所有线段到DXF文档，包括y坐标翻转
            for line in self.linesegs.linesegments:
                ax, ay = float(line.point_a[0]), float(line.point_a[1])
                bx, by = float(line.point_b[0]), float(line.point_b[1])
                
                # 翻转y坐标以匹配PNG图片的显示方向
                if img_height > 0:
                    ay = img_height - ay
                    by = img_height - by
                
                msp.add_line((ax, ay, 0.0), (bx, by, 0.0))
            
            # 保存DXF文件
            doc.saveas(file_path)
            return True
        except Exception as e:
            print(f"保存DXF文件失败: {str(e)}")
            return False
    
    def on_save_image(self):
        """处理保存图片按钮点击事件"""
        # 打开文件保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存图片",
            "",
            "图片文件 (*.png *.jpg *.bmp);;PNG文件 (*.png);;JPG文件 (*.jpg);;BMP文件 (*.bmp)"
        )
        
        if file_path:
            # 确保文件有扩展名
            if not any(file_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp']):
                file_path += '.png'  # 默认保存为PNG格式
            
            # 调用保存图片方法
            if self.save_image(file_path):
                print(f"图片已成功保存到: {file_path}")
            else:
                print(f"图片保存失败")
    
    def on_save_dxf(self):
        """处理保存DXF文件按钮点击事件"""
        # 打开文件保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存DXF文件",
            "",
            "DXF文件 (*.dxf)"
        )
        
        if file_path:
            # 确保文件有.dxf扩展名
            if not file_path.lower().endswith('.dxf'):
                file_path += '.dxf'
            
            # 调用保存DXF方法
            if self.save_dxf(file_path):
                print(f"DXF文件已成功保存到: {file_path}")
            else:
                print(f"DXF文件保存失败")

    
    def update_line_properties(self, line):
        """更新线段属性显示
        
        Args:
            line: LineSegment对象
        """
        if line:
            # 检查属性面板组件是否已初始化
            if hasattr(self, 'point_a_x') and self.point_a_x is not None:
                self.point_a_x.blockSignals(True)
                self.point_a_y.blockSignals(True)
                self.point_b_x.blockSignals(True)
                self.point_b_y.blockSignals(True)
                
                self.point_a_x.setValue(line.point_a[0])
                self.point_a_y.setValue(line.point_a[1])
                self.point_b_x.setValue(line.point_b[0])
                self.point_b_y.setValue(line.point_b[1])
                
                # 计算线段长度
                length = np.sqrt((line.point_b[0] - line.point_a[0])**2 + (line.point_b[1] - line.point_a[1])** 2)
                if hasattr(self, 'line_length') and self.line_length is not None:
                    self.line_length.setText(f"{length:.4f}")
                # 更新线段对象的length属性
                line.length = length
                
                self.point_a_x.blockSignals(False)
                self.point_a_y.blockSignals(False)
                self.point_b_x.blockSignals(False)
                self.point_b_y.blockSignals(False)
    
    def update_selected_line_info(self):
        """更新选中线段的信息"""
        if self.canvas.selected_line_idx >= 0 and self.linesegs and self.canvas.selected_line_idx < len(self.linesegs.linesegments):
            line = self.linesegs.linesegments[self.canvas.selected_line_idx]
            self.update_line_properties(line)
            self.lineSelected.emit(line)
            
            # 更新线段列表中的选中状态
            self.selected_line_id = self.canvas.selected_line_idx
            for item in self.line_items:
                item.set_selected(item.line_id == self.selected_line_id)
    
    def clear_selected_line_info(self):
        """清除选中线段的信息"""
        # 检查属性面板组件是否已初始化
        if hasattr(self, 'point_a_x') and self.point_a_x is not None:
            self.point_a_x.blockSignals(True)
            self.point_a_y.blockSignals(True)
            self.point_b_x.blockSignals(True)
            self.point_b_y.blockSignals(True)
            
            self.point_a_x.setValue(0.0)
            self.point_a_y.setValue(0.0)
            self.point_b_x.setValue(0.0)
            self.point_b_y.setValue(0.0)
            
            if hasattr(self, 'line_length') and self.line_length is not None:
                self.line_length.setText("0.0000")
            
            self.point_a_x.blockSignals(False)
            self.point_a_y.blockSignals(False)
            self.point_b_x.blockSignals(False)
            self.point_b_y.blockSignals(False)
        
        # 清除线段列表中的选中状态
        self.selected_line_id = -1
        for item in self.line_items:
            item.set_selected(False)
    
    def on_point_moved(self, line_idx, point_idx, new_pos):
        """处理端点移动事件
        
        Args:
            line_idx: 线段索引
            point_idx: 端点索引 (0: point_a, 1: point_b)
            new_pos: 新的坐标位置
        """
        if not self.linesegs or line_idx >= len(self.linesegs.linesegments):
            return
        
        line = self.linesegs.linesegments[line_idx]
        # 更新线段端点位置
        if point_idx == 0:
            line.point_a = np.array(new_pos)
        else:
            line.point_b = np.array(new_pos)
        
        # 重新计算线段长度和方向
        line.direction = line.point_b - line.point_a
        line.direction = line.direction.astype(float)
        line.direction /= np.linalg.norm(line.direction)
        line.length = np.linalg.norm(line.point_b - line.point_a)
        
        # 发出修改信号
        self.lineModified.emit(line)
        
        # 更新线段列表，确保上方显示的长度也同步更新
        self.update_line_list()
        
        # 如果当前移动的线段是选中的，更新属性面板
        if self.canvas.selected_line_idx == line_idx:
            self.update_selected_line_info()
    
    def on_drag_finished(self, line_idx):
        """处理拖拽结束事件
        
        Args:
            line_idx: 完成拖拽的线段索引
        """
        if line_idx == self.canvas.selected_line_idx and line_idx < len(self.linesegs.linesegments):
            # 只有在拖拽结束时才更新属性面板
            line = self.linesegs.linesegments[line_idx]
            self.update_line_properties(line)
    
    def on_point_a_x_changed(self, value):
        """处理端点A x坐标变化"""
        if self.canvas.selected_line_idx >= 0 and self.linesegs and self.canvas.selected_line_idx < len(self.linesegs.linesegments):
            line = self.linesegs.linesegments[self.canvas.selected_line_idx]
            line.point_a = (value, line.point_a[1])
            # 更新线段对象的长度属性
            line.length = line.get_length()
            # 更新方向向量
            line.direction = np.array(line.point_b) - np.array(line.point_a)
            if np.linalg.norm(line.direction) > 0:
                line.direction /= np.linalg.norm(line.direction)
            self.canvas.update()
            # 更新属性面板
            self.update_line_properties(line)
            # 更新线段列表（包括长度信息）
            self.update_line_list()
            # 发送修改信号
            self.lineModified.emit(line)
    
    def on_point_a_y_changed(self, value):
        """处理端点A y坐标变化"""
        if self.canvas.selected_line_idx >= 0 and self.linesegs and self.canvas.selected_line_idx < len(self.linesegs.linesegments):
            line = self.linesegs.linesegments[self.canvas.selected_line_idx]
            line.point_a = (line.point_a[0], value)
            # 更新线段对象的长度属性
            line.length = line.get_length()
            # 更新方向向量
            line.direction = np.array(line.point_b) - np.array(line.point_a)
            if np.linalg.norm(line.direction) > 0:
                line.direction /= np.linalg.norm(line.direction)
            self.canvas.update()
            # 更新属性面板
            self.update_line_properties(line)
            # 更新线段列表（包括长度信息）
            self.update_line_list()
            # 发送修改信号
            self.lineModified.emit(line)
    
    def on_point_b_x_changed(self, value):
        """处理端点B x坐标变化"""
        if self.canvas.selected_line_idx >= 0 and self.linesegs and self.canvas.selected_line_idx < len(self.linesegs.linesegments):
            line = self.linesegs.linesegments[self.canvas.selected_line_idx]
            line.point_b = (value, line.point_b[1])
            # 更新线段对象的长度属性
            line.length = line.get_length()
            # 更新方向向量
            line.direction = np.array(line.point_b) - np.array(line.point_a)
            if np.linalg.norm(line.direction) > 0:
                line.direction /= np.linalg.norm(line.direction)
            self.canvas.update()
            # 更新属性面板
            self.update_line_properties(line)
            # 更新线段列表（包括长度信息）
            self.update_line_list()
            # 发送修改信号
            self.lineModified.emit(line)
    
    def on_point_b_y_changed(self, value):
        """处理端点B y坐标变化"""
        if self.canvas.selected_line_idx >= 0 and self.linesegs and self.canvas.selected_line_idx < len(self.linesegs.linesegments):
            line = self.linesegs.linesegments[self.canvas.selected_line_idx]
            line.point_b = (line.point_b[0], value)
            # 更新线段对象的长度属性
            line.length = line.get_length()
            # 更新方向向量
            line.direction = np.array(line.point_b) - np.array(line.point_a)
            if np.linalg.norm(line.direction) > 0:
                line.direction /= np.linalg.norm(line.direction)
            self.canvas.update()
            # 更新属性面板
            self.update_line_properties(line)
            # 更新线段列表（包括长度信息）
            self.update_line_list()
            # 发送修改信号
            self.lineModified.emit(line)
    
    def delete_line(self):
        """删除选中的线段"""
        if self.canvas.selected_line_idx >= 0 and self.linesegs and self.canvas.selected_line_idx < len(self.linesegs.linesegments):
            # 记录要删除的线段索引
            deleted_idx = self.canvas.selected_line_idx
            # 从线段集中移除线段
            self.linesegs.linesegments.pop(deleted_idx)
            # 更新画布
            self.canvas.selected_line_idx = -1
            self.canvas.selected_point_idx = -1
            self.canvas.update()
            # 清除属性面板
            self.clear_selected_line_info()
            # 发出删除信号
            self.lineDeleted.emit(deleted_idx)
            
            # 更新线段列表
            self.update_line_list()
    
    def from_data_source(self, img=None, linesegs=None):
        """从数据源加载数据
        
        Args:
            img: numpy格式的图像
            linesegs: LineSegments对象
        """
        # 设置图像
        if img is not None:
            self.set_image(img)
        
        # 设置线段集
        if linesegs is not None:
            self.set_linesegs(linesegs)
    
    def get_linesegs(self):
        """获取当前的线段集
        
        Returns:
            LineSegments对象
        """
        return self.linesegs
    
    def get_image(self):
        """获取当前的图像
        
        Returns:
            numpy数组格式的图像
        """
        return self.img.copy()  # 返回副本以避免外部修改


class LineEditorCanvas(QWidget):
    """线段编辑画布"""
    pointMoved = pyqtSignal(int, int, tuple)  # 发送：线段索引，端点索引，新坐标
    dragFinished = pyqtSignal(int)  # 发送：线段索引，表示拖拽结束
    
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
        # 缓存缩放信息以提高性能
        self.last_scale_x = 1.0
        self.last_scale_y = 1.0
        self.last_offset_x = 0
        self.last_offset_y = 0
        # 设置鼠标跟踪，确保即使不按下鼠标也能接收moveEvent
        self.setMouseTracking(True)
        # 设置背景颜色
        self.setStyleSheet("background-color: #1e1e1e; border: none;")
    
    def setParent(self, parent):
        """设置父对象"""
        super().setParent(parent)
        self.parent_widget = parent
        
    def set_image(self, img):
        """设置背景图像
        
        Args:
            img: numpy格式的二维图像数组，支持灰度图和RGB图像
        """
        # 验证输入是否为numpy数组
        if not isinstance(img, np.ndarray):
            raise TypeError("Input image must be a numpy array")
        
        # 验证图像维度是否正确（灰度图或RGB图）
        if len(img.shape) not in [2, 3]:
            raise ValueError("Input image must be 2D (grayscale) or 3D (RGB/BGR)")
        
        # 存储图像
        self.img = img
        
        # 更新pixmap并刷新画布
        self._update_pixmap()
        self.update()
    
    def set_linesegs(self, linesegs):
        """设置线段集
        
        Args:
            linesegs: LineSegments对象，包含多个LineSegment
        """
        # 验证输入是否为LineSegments对象
        if not hasattr(linesegs, 'linesegments'):
            raise TypeError("Input must be a LineSegments object with linesegments attribute")
        
        # 验证linesegments是否为列表
        if not isinstance(linesegs.linesegments, list):
            raise TypeError("linesegments attribute must be a list")
        
        # 验证列表中的每个元素是否为LineSegment类型（检查是否具有point_a和point_b属性）
        for line in linesegs.linesegments:
            if not hasattr(line, 'point_a') or not hasattr(line, 'point_b'):
                raise TypeError("Each line must be a LineSegment object with point_a and point_b attributes")
        
        # 存储线段集
        self.linesegs = linesegs
        
        # 重置选中状态
        self.selected_line_idx = -1
        self.selected_point_idx = -1
        
        # 刷新画布
        self.update()
    
    def _update_pixmap(self):
        """将numpy图像转换为QPixmap"""
        if self.img is None:
            return
        
        height, width = self.img.shape[:2]
        
        # 如果是灰度图，转换为RGB
        if len(self.img.shape) == 2:
            bytes_per_line = width
            q_img = QImage(self.img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            # BGR -> RGB
            rgb_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            bytes_per_line = width * 3
            q_img = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
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
    
    def _calculate_scale_and_offset(self):
        """计算并缓存缩放比例和偏移量"""
        if self.img_pixmap and self.img is not None:
            scaled_pixmap = self.img_pixmap.scaled(
                self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.last_offset_x = (self.width() - scaled_pixmap.width()) // 2
            self.last_offset_y = (self.height() - scaled_pixmap.height()) // 2
            self.last_scale_x = scaled_pixmap.width() / self.img.shape[1]
            self.last_scale_y = scaled_pixmap.height() / self.img.shape[0]
            return self.last_scale_x, self.last_scale_y, self.last_offset_x, self.last_offset_y
        return 1.0, 1.0, 0, 0
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if self.img is None or self.linesegs is None:
            return
        
        # 获取鼠标坐标
        mouse_x = event.x()
        mouse_y = event.y()
        
        # 计算缩放比例和偏移
        if self.img_pixmap:
            # 计算并缓存缩放信息
            scale_x, scale_y, offset_x, offset_y = self._calculate_scale_and_offset()
            
            # 检查是否点击了某个端点或线段
            clicked = False
            min_dist = float('inf')
            closest_line_idx = -1
            closest_point_idx = -1
            
            # 首先检查是否点击了端点
            for i, line in enumerate(self.linesegs.linesegments):
                # 端点A
                x1 = int(line.point_a[0] * scale_x + offset_x)
                y1 = int(line.point_a[1] * scale_y + offset_y)
                dist = np.sqrt((mouse_x - x1)** 2 + (mouse_y - y1)** 2)
                if dist < min_dist and dist <= self.point_radius:
                    min_dist = dist
                    closest_line_idx = i
                    closest_point_idx = 0
                    clicked = True
                
                # 端点B
                x2 = int(line.point_b[0] * scale_x + offset_x)
                y2 = int(line.point_b[1] * scale_y + offset_y)
                dist = np.sqrt((mouse_x - x2)** 2 + (mouse_y - y2)** 2)
                if dist < min_dist and dist <= self.point_radius:
                    min_dist = dist
                    closest_line_idx = i
                    closest_point_idx = 1
                    clicked = True
            
            # 如果没有点击端点，检查是否点击了线段
            if not clicked:
                for i, line in enumerate(self.linesegs.linesegments):
                    x1 = int(line.point_a[0] * scale_x + offset_x)
                    y1 = int(line.point_a[1] * scale_y + offset_y)
                    x2 = int(line.point_b[0] * scale_x + offset_x)
                    y2 = int(line.point_b[1] * scale_y + offset_y)
                    
                    dist = self._point_to_line_distance(mouse_x, mouse_y, x1, y1, x2, y2)
                    if dist < min_dist and dist <= 5:  # 5像素的容差
                        min_dist = dist
                        closest_line_idx = i
                        closest_point_idx = -1
                        clicked = True
            
            # 处理点击事件
            if clicked:
                # 更新选中状态
                self.selected_line_idx = closest_line_idx
                self.selected_point_idx = closest_point_idx
                self.is_dragging = (closest_point_idx != -1)
                self.drag_start_pos = QPoint(mouse_x, mouse_y)
                
                # 计算点击偏移量
                if self.is_dragging:
                    line = self.linesegs.linesegments[closest_line_idx]
                    if closest_point_idx == 0:
                        point_x = line.point_a[0] * scale_x + offset_x
                        point_y = line.point_a[1] * scale_y + offset_y
                    else:
                        point_x = line.point_b[0] * scale_x + offset_x
                        point_y = line.point_b[1] * scale_y + offset_y
                    self.click_offset = (mouse_x - point_x, mouse_y - point_y)
                
                # 确保调用父组件的更新方法来刷新属性面板
                if hasattr(self.parent_widget, 'update_selected_line_info'):
                    self.parent_widget.update_selected_line_info()
                    
                # 发送线段选中信号
                if hasattr(self.parent_widget, 'lineSelected'):
                    line = self.linesegs.linesegments[closest_line_idx]
                    self.parent_widget.lineSelected.emit(line)
            else:
                # 没有点击任何线段，清除选中状态
                self.selected_line_idx = -1
                self.selected_point_idx = -1
                if hasattr(self.parent_widget, 'clear_selected_line_info'):
                    self.parent_widget.clear_selected_line_info()
            
            # 更新画布
            self.update()
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.img is None or self.linesegs is None:
            return
        
        # 如果正在拖动端点
        if self.is_dragging and self.selected_line_idx != -1 and self.selected_point_idx != -1:
            # 获取鼠标坐标
            mouse_x = event.x()
            mouse_y = event.y()
            
            # 使用缓存的缩放比例和偏移量，避免重复计算
            # 计算新的端点位置
            new_x = (mouse_x - self.last_offset_x - self.click_offset[0]) / self.last_scale_x
            new_y = (mouse_y - self.last_offset_y - self.click_offset[1]) / self.last_scale_y
            
            # 发出位置移动信号
            self.pointMoved.emit(self.selected_line_idx, self.selected_point_idx, (new_x, new_y))
            
            # 直接更新线段端点并刷新画布，避免通过信号链导致的延迟
            line = self.linesegs.linesegments[self.selected_line_idx]
            if self.selected_point_idx == 0:
                line.point_a = (new_x, new_y)
            else:
                line.point_b = (new_x, new_y)
            self.update()  # 立即更新画布
        else:
            # 更新悬停状态
            self._update_hovered_line(event)
    
    def _update_hovered_line(self, event):
        """更新悬停的线段"""
        if self.img is None or self.linesegs is None or self.is_dragging:
            return
        
        # 获取鼠标坐标
        mouse_x = event.x()
        mouse_y = event.y()
        
        # 计算缩放比例和偏移
        scaled_pixmap = self.img_pixmap.scaled(
            self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        offset_x = (self.width() - scaled_pixmap.width()) // 2
        offset_y = (self.height() - scaled_pixmap.height()) // 2
        
        scale_x = scaled_pixmap.width() / self.img.shape[1]
        scale_y = scaled_pixmap.height() / self.img.shape[0]
        
        # 检查是否悬停在端点上
        hovered_line_idx = -1
        min_dist = float('inf')
        
        for i, line in enumerate(self.linesegs.linesegments):
            if i == self.selected_line_idx:
                continue  # 跳过选中的线段
            
            # 端点A
            x1 = int(line.point_a[0] * scale_x + offset_x)
            y1 = int(line.point_a[1] * scale_y + offset_y)
            dist = np.sqrt((mouse_x - x1)** 2 + (mouse_y - y1)** 2)
            if dist < min_dist and dist <= self.point_radius:
                min_dist = dist
                hovered_line_idx = i
            
            # 端点B
            x2 = int(line.point_b[0] * scale_x + offset_x)
            y2 = int(line.point_b[1] * scale_y + offset_y)
            dist = np.sqrt((mouse_x - x2)** 2 + (mouse_y - y2)** 2)
            if dist < min_dist and dist <= self.point_radius:
                min_dist = dist
                hovered_line_idx = i
            
            # 线段本身
            dist = self._point_to_line_distance(mouse_x, mouse_y, x1, y1, x2, y2)
            if dist < min_dist and dist <= 5:  # 5像素的容差
                min_dist = dist
                hovered_line_idx = i
        
        # 更新悬停状态
        if hovered_line_idx != self.hovered_line_idx:
            self.hovered_line_idx = hovered_line_idx
            self.update()
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if self.is_dragging and self.selected_line_idx != -1:
            # 拖拽结束时发出信号，用于更新属性面板
            self.dragFinished.emit(self.selected_line_idx)
        self.is_dragging = False
    
    def save_image(self, file_path):
        """保存画布内容为图片，只包含地图和红色线段，忽略高亮和端点
        
        Args:
            file_path: 保存的文件路径
        
        Returns:
            bool: 保存是否成功
        """
        try:
            # 确保图像存在
            if self.img_pixmap is None:
                print("没有图像可保存")
                return False
                
            # 确保目录存在
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # 创建一个新的图像，只包含地图和红色线段
            # 计算缩放比例和偏移
            scaled_pixmap = self.img_pixmap.scaled(
                self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            # 创建一个与缩放后图像相同大小的QImage
            result_img = QImage(scaled_pixmap.size(), QImage.Format_RGB32)
            result_img.fill(Qt.white)  # 白色背景
            
            # 创建QPainter对象用于绘制
            painter = QPainter(result_img)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # 绘制背景图像
            painter.drawPixmap(0, 0, scaled_pixmap)
            
            # 绘制所有线段，但只使用红色，不绘制端点
            if self.linesegs:
                # 设置红色画笔
                pen = QPen(QColor(255, 0, 0), 2)  # 红色，2像素宽
                painter.setPen(pen)
                
                # 计算缩放比例
                scale_x = scaled_pixmap.width() / self.img.shape[1]
                scale_y = scaled_pixmap.height() / self.img.shape[0]
                
                # 绘制所有线段，忽略高亮和端点
                for line in self.linesegs.linesegments:
                    # 计算线段端点坐标
                    x1 = int(line.point_a[0] * scale_x)
                    y1 = int(line.point_a[1] * scale_y)
                    x2 = int(line.point_b[0] * scale_x)
                    y2 = int(line.point_b[1] * scale_y)
                    
                    # 绘制线段（不绘制端点）
                    painter.drawLine(x1, y1, x2, y2)
            
            # 结束绘制
            painter.end()
            
            # 保存图像
            return result_img.save(file_path)
        except Exception as e:
            print(f"保存图片失败: {str(e)}")
            return False
            
    def _point_to_line_distance(self, x, y, x1, y1, x2, y2):
        """计算点到线段的距离
        
        Args:
            x, y: 点的坐标
            x1, y1: 线段端点1的坐标
            x2, y2: 线段端点2的坐标
            
        Returns:
            点到线段的距离
        """
        # 线段的向量
        line_vec = np.array([x2 - x1, y2 - y1])
        # 线段长度的平方
        line_len_sq = line_vec.dot(line_vec)
        
        # 如果线段长度为0，直接返回点到端点的距离
        if line_len_sq == 0:
            return np.sqrt((x - x1)** 2 + (y - y1)** 2)
        
        # 计算投影参数t
        t = max(0, min(1, np.array([x - x1, y - y1]).dot(line_vec) / line_len_sq))
        
        # 计算投影点
        projection = np.array([x1 + t * line_vec[0], y1 + t * line_vec[1]])
        
        # 计算点到投影点的距离
        return np.sqrt((x - projection[0])** 2 + (y - projection[1])** 2)

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt

class TestWindow(QMainWindow):
        def __init__(self,img=None,linesegs=None):
            super().__init__()
            self.setWindowTitle("LiDAR Point Cloud Studio v1.0")
            self.setGeometry(100, 100, 1200, 800)
            #提供有输入线段和无输入线段的两种展示方式
            self.img=None
            self.linesegs=None
            # 创建线段编辑器组件
            self.editor = LineEditorComponent()
            self.setCentralWidget(self.editor)
            
            # 连接信号
            self.editor.lineSelected.connect(self.on_line_selected)
            self.editor.lineModified.connect(self.on_line_modified)
            self.editor.lineDeleted.connect(self.on_line_deleted)
            
            if img is None or linesegs is None:
                # 创建测试图像和线段
                self._create_test_data()
            else:
                self.set_data(img,linesegs)
        
        def _create_test_data(self):
            """创建测试数据，模拟真实的LiDAR扫描结果"""
            # 创建一个测试图像（模拟点云投影）
            print("当前没有传入图像和线段,创建测试数据")
            img1 = np.zeros((600, 800), dtype=np.uint8)
            # 绘制一些建筑物轮廓
            cv2.rectangle(img1, (100, 100), (400, 400), 255, 1)
            cv2.rectangle(img1, (450, 150), (600, 350), 255, 1)
            cv2.line(img1, (100, 400), (400, 400), 255, 1)
            cv2.line(img1, (450, 350), (600, 350), 255, 1)
            # 绘制一些窗户和门的线条
            for i in range(150, 350, 50):
                cv2.line(img1, (i, 100), (i, 400), 128, 1)
            # 创建线段（模拟检测到的线段）- 使用numpy数组
            line1 = LineSegment(np.array([100, 100]), np.array([400, 100]))  # 屋顶线
            line2 = LineSegment(np.array([100, 100]), np.array([100, 400]))  # 左边线
            line3 = LineSegment(np.array([400, 100]), np.array([400, 400]))  # 右边线
            line4 = LineSegment(np.array([100, 400]), np.array([400, 400]))  # 底线
            line5 = LineSegment(np.array([450, 150]), np.array([600, 150]))  # 小屋顶线
            line6 = LineSegment(np.array([450, 150]), np.array([450, 350]))  # 小左边线
            line7 = LineSegment(np.array([600, 150]), np.array([600, 350]))  # 小右边线
            line8 = LineSegment(np.array([450, 350]), np.array([600, 350]))  # 小底线
            
            # 创建线段集 - 直接传入线段列表
            linesegs1 = LineSegments([line1, line2, line3, line4, line5, line6, line7, line8])
            self.img=img1
            self.linesegs=linesegs1
        
            # 设置到编辑器
            self.editor.set_image(self.img)
            self.editor.set_linesegs(self.linesegs)
            
        def set_data(self,img,linesegs):
            """设置数据
            print("当前已传入图像和线段,创建测试数据")
            Args:
                img: 输入图像
                linesegs: 输入线段集
            """
            self.img=img
            self.linesegs=linesegs
            # 设置到编辑器
            self.editor.set_image(self.img)
            self.editor.set_linesegs(self.linesegs)
        
        def on_line_selected(self, line):
            """处理线段选中事件"""
            print(f"选中线段: 端点A={line.point_a}, 端点B={line.point_b}")
        
        def on_line_modified(self, line):
            """处理线段修改事件"""
            print(f"修改线段: 端点A={line.point_a}, 端点B={line.point_b}")
        
        def on_line_deleted(self, line_idx):
            """处理线段删除事件"""
            print(f"删除线段索引: {line_idx}")

# 测试代码
if __name__ == "__main__":
    
    
    app = QApplication(sys.argv)
    # 设置全局样式表，更接近LiDAR应用的暗色主题
    app.setStyleSheet("""
        QWidget {
            background-color: #2d2d2d;
            color: #ffffff;
            font-family: Arial, sans-serif;
        }
        QMainWindow, QWidget {
            border: none;
        }
    """)
    window = TestWindow(img=None,linesegs=None)
    window.show()
    sys.exit(app.exec_())