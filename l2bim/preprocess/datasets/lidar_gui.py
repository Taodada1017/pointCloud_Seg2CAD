import sys
import os
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QLabel,
                            QVBoxLayout, QHBoxLayout, QGroupBox, QLineEdit, QFrame,
                            QFileDialog, QGridLayout, QDialog, QSplitter, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon

# 线段编辑功能已移至line_editor_qt.py脚本

class MainPage(QWidget):
    """主页面基类"""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        # 存储参数默认值，用于重置功能
        self.param_defaults_preview = [
            ("方位角(Theta):", 0.0),
            ("仰角(Phi):", 0.0),
            ("观察距离:", 2.0),
            ("背景色:", 0.0),
            ("下采样大小:", 0.1)
        ]

        # 点云投影页面参数默认值
        self.param_defaults_projection = [
            ("线条粗细:", 30.0)
        ]
        # 直接检测默认参数值
        self.param_defaults_detect = [
            ("检测最短线段长度:", 30.0)
        ]
        # 存储参数控件引用
        self.param_spin_boxes_preview = []
        self.param_spin_boxes_projection = []
        self.param_spin_boxes_detect = []
        self.initUI()
    
    def initUI(self):
        # 主布局
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(15)  # 增加主布局间距
        main_layout.setContentsMargins(10, 10, 10, 10)  # 设置边距
        
        # 左侧控制面板 - 统一宽度
        control_panel = QWidget()
        control_panel.setMaximumWidth(300)  # 统一设置为300px宽度
        control_panel.setMinimumWidth(300)  # 确保最小宽度一致
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(12)  # 增加内部间距
        control_layout.setContentsMargins(10, 10, 10, 10)
        
        # 文件选择组
        file_group = QGroupBox("文件选择")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(10)
        
        # PCD文件选择按钮  这里也要修改，只有点云预览有这个pcd选择按钮
        if self.title not in ["点云投影","直接检测"]:
            self.pcd_file_button = QPushButton("请选择PCD文件")
            self.pcd_file_button.clicked.connect(self.select_pcd_file)
            self.pcd_file_button.setMinimumHeight(40)  # 增加按钮高度确保文字完全显示
            self.pcd_file_button.setStyleSheet("""
                QPushButton {
                    background-color: #555555;
                    color: #CCCCCC;
                    font-size: 14px;
                    font-family: 'SimHei', 'Microsoft YaHei', sans-serif;
                    border: 1px solid #666666;
                    text-align: center;
                    padding: 5px;
                    white-space: normal;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background-color: #666666;
                }
                QPushButton:pressed {
                    background-color: #333333;
                }
            """)
            file_layout.addWidget(self.pcd_file_button)
        
        # 配置文件选择（只有点云预览界面需要）
        if self.title not in ["点云投影","直接检测"]:   #这里改成别的就不会触发
            self.config_file_button = QPushButton("请选择yaml文件")
            self.config_file_button.clicked.connect(self.select_config_file)
            self.config_file_button.setMinimumHeight(40)  # 增加按钮高度确保文字完全显示
            self.config_file_button.setStyleSheet("""
                QPushButton {
                    background-color: #555555;
                    color: #CCCCCC;
                    font-size: 14px;
                    font-family: 'SimHei', 'Microsoft YaHei', sans-serif;
                    border: 1px solid #666666;
                    text-align: center;
                    padding: 5px;
                    white-space: normal;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background-color: #666666;
                }
                QPushButton:pressed {
                    background-color: #333333;
                }
            """)
            file_layout.addWidget(self.config_file_button)
        
        #这里也进行修改，只在点云预览界面出现文件选择框
        if self.title not in ["点云投影","直接检测"]:   
            control_layout.addWidget(file_group)
        
        # 仅点云预览页面有参数设置
        if self.title == "点云预览":
            params_group = QGroupBox("参数设置")
            params_layout = QVBoxLayout(params_group)
            params_layout.setSpacing(8)
            
            # 视角参数
            angle_params = [
                ("方位角(Theta):", 0.0, -360.0, 360.0, 0.1),
                ("仰角(Phi):", 0.0, -90.0, 90.0, 0.1),
                ("观察距离:", 2.0, 0.1, 100.0, 0.1),
                ("背景色:", 0.0, 0.0, 1.0, 0.1),
                ("下采样大小:", 0.1, 0.0, 1.0, 0.1)
            ]
            
            for label_text, default_value, min_val, max_val, step in angle_params:
                param_layout = QHBoxLayout()
                param_layout.setSpacing(8)
                label = QLabel(label_text)
                label.setMinimumWidth(120)  # 增加标签宽度，确保文字完全显示
                label.setMaximumWidth(120)
                label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # 右对齐
                label.setStyleSheet("font-size: 14px; font-family: 'SimHei', 'Microsoft YaHei', sans-serif; color: #FFFFFF;")
                spin_box = QDoubleSpinBox()
                spin_box.setDecimals(4)  # 设置小数位数
                spin_box.setRange(min_val, max_val)  # 设置范围
                spin_box.setSingleStep(step)  # 设置步长
                spin_box.setValue(float(default_value))  # 设置默认值
                spin_box.setMinimumWidth(120)  # 增加宽度
                spin_box.setStyleSheet("""
                    QDoubleSpinBox {
                        background-color: #444444;
                        border: 1px solid #666666;
                        padding: 6px 8px;
                        color: #FFFFFF;
                        font-size: 14px;
                        min-height: 28px;
                    }
                    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                        width: 20px;
                    }
                """)
                param_layout.addWidget(label)
                param_layout.addWidget(spin_box)
                params_layout.addLayout(param_layout)
                params_layout.addSpacing(5)  # 增加每个参数之间的间距
                # 保存参数控件引用，用于重置功能
                self.param_spin_boxes_preview.append(spin_box)
            
            control_layout.addWidget(params_group)
        
        elif self.title == "点云投影":
            params_group = QGroupBox("参数设置")
            params_layout = QVBoxLayout(params_group)
            params_layout.setSpacing(8)
            
            # 视角参数
            angle_params = [
                ("线条粗细(Scale):", 10.0, 1.0, 300.0, 1.0)
            ]
            
            for label_text, default_value, min_val, max_val, step in angle_params:
                param_layout = QHBoxLayout()
                param_layout.setSpacing(8)
                label = QLabel(label_text)
                label.setMinimumWidth(120)  # 增加标签宽度，确保文字完全显示
                label.setMaximumWidth(120)
                label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # 右对齐
                label.setStyleSheet("font-size: 14px; font-family: 'SimHei', 'Microsoft YaHei', sans-serif; color: #FFFFFF;")
                spin_box = QDoubleSpinBox()
                spin_box.setDecimals(4)  # 设置小数位数
                spin_box.setRange(min_val, max_val)  # 设置范围
                spin_box.setSingleStep(step)  # 设置步长
                spin_box.setValue(float(default_value))  # 设置默认值
                spin_box.setMinimumWidth(120)  # 增加宽度
                spin_box.setStyleSheet("""
                    QDoubleSpinBox {
                        background-color: #444444;
                        border: 1px solid #666666;
                        padding: 6px 8px;
                        color: #FFFFFF;
                        font-size: 14px;
                        min-height: 28px;
                    }
                    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                        width: 20px;
                    }
                """)
                param_layout.addWidget(label)
                param_layout.addWidget(spin_box)
                params_layout.addLayout(param_layout)
                params_layout.addSpacing(5)  # 增加每个参数之间的间距
                # 保存参数控件引用，用于重置功能
                self.param_spin_boxes_projection.append(spin_box)
            
            control_layout.addWidget(params_group)

        elif self.title == "直接检测":
            params_group = QGroupBox("参数设置")
            params_layout = QVBoxLayout(params_group)
            params_layout.setSpacing(8)
            
            # 视角参数
            angle_params = [
                ("检测最短线段长度:", 30.0, 20.0, 100.0, 1.0)
            ]
            
            for label_text, default_value, min_val, max_val, step in angle_params:
                param_layout = QHBoxLayout()
                param_layout.setSpacing(8)
                label = QLabel(label_text)
                label.setMinimumWidth(120)  # 增加标签宽度，确保文字完全显示
                label.setMaximumWidth(120)
                label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # 右对齐
                label.setStyleSheet("font-size: 14px; font-family: 'SimHei', 'Microsoft YaHei', sans-serif; color: #FFFFFF;")
                spin_box = QDoubleSpinBox()
                spin_box.setDecimals(4)  # 设置小数位数
                spin_box.setRange(min_val, max_val)  # 设置范围
                spin_box.setSingleStep(step)  # 设置步长
                spin_box.setValue(float(default_value))  # 设置默认值
                spin_box.setMinimumWidth(120)  # 增加宽度
                spin_box.setStyleSheet("""
                    QDoubleSpinBox {
                        background-color: #444444;
                        border: 1px solid #666666;
                        padding: 6px 8px;
                        color: #FFFFFF;
                        font-size: 14px;
                        min-height: 28px;
                    }
                    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                        width: 20px;
                    }
                """)
                param_layout.addWidget(label)
                param_layout.addWidget(spin_box)
                params_layout.addLayout(param_layout)
                params_layout.addSpacing(5)  # 增加每个参数之间的间距
                # 保存参数控件引用，用于重置功能
                self.param_spin_boxes_detect.append(spin_box)
            
            control_layout.addWidget(params_group)
        
        # 生成预览/投影/检测按钮
        if "预览" in self.title:
            action_text="生成预览"
        elif "投影" in self.title:
            action_text="生成投影" 
        elif "检测" in self.title:
            action_text="生成线段"
        # action_text = "生成预览" if "预览" in self.title else "生成投影"
        # action_text = "生成线段" if "检测" in self.title else "生成投影"
        self.generate_button = QPushButton(action_text)
        self.generate_button.setStyleSheet("""
            background-color: #006400;
            color: white;
            min-height: 40px;
            font-size: 14px;
            font-weight: bold;
        """)
        # 添加按动效果
        if "检测" in self.title or "投影" in self.title:
            margin="55px"
        elif "预览" in self.title:
            margin="5px"
        self.generate_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #006400;
                color: white;
                min-height: 40px;
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #008000;
                margin-top: {margin};  /* 设置距离上边界的距离为margin像素 */
            }}
            QPushButton:hover {{
                background-color: #007D00;
            }}
            QPushButton:pressed {{
                background-color: #004D00;
                border-color: #003300;
            }}
        """)
        
        # 重置参数按钮
        self.reset_button = QPushButton("重置参数")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: #CCCCCC;
                min-height: 32px;
                font-size: 14px;
                border: 1px solid #666666;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QPushButton:pressed {
                background-color: #333333;
                border-color: #444444;
            }
        """)
        # 连接重置按钮的点击事件
        self.reset_button.clicked.connect(self.reset_parameters)
        
        control_layout.addWidget(self.generate_button)
        control_layout.addSpacing(8)
        control_layout.addWidget(self.reset_button)
        control_layout.addSpacing(8)
        
        # 仅检测页面有检测结果
        if "检测" in self.title:
            results_group = QGroupBox("线段信息")  # 修改为线段信息，与用户要求一致
            results_layout = QVBoxLayout(results_group)
            
            # 创建空白的检测结果区域，等待后续接口集成
            results_text = QTextEdit()
            results_text.setReadOnly(True)
            results_text.setPlaceholderText("等待检测结果...")  # 使用占位符文本
            results_text.setText("")  # 保持空白状态
            results_layout.addWidget(results_text)
            
            control_layout.addWidget(results_group)
        
        control_layout.addStretch()
        
        # 右侧预览区域
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setSpacing(12)  # 增加预览区域间距
        preview_layout.setContentsMargins(10, 10, 10, 10)
        
        # 预览区域标题
        preview_title = QLabel(self.title)
        preview_title.setAlignment(Qt.AlignCenter)
        preview_title.setStyleSheet("font-size: 18px; font-weight: bold; color: white; padding: 10px;")
        
        # 预览画布
        self.canvas_widget = QWidget()
        self.canvas_widget.setStyleSheet("background-color: #444444; border: 1px solid #666666;")
        self.canvas_widget.setMinimumHeight(800)
        
        # 默认提示文本
        if "预览" in self.title:
            default_text = "请选择PCD文件并点击生成预览"
        elif "投影" in self.title:
            default_text = "请选择PCD文件并点击生成投影"
        elif "直接检测" in self.title:
            default_text = "请选择PCD文件并点击直接检测"
        self.hint_label = QLabel(default_text)
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setStyleSheet("color: #AAAAAA; font-size: 14px; padding: 10px;")
        self.hint_label.setWordWrap(True)  # 允许文本换行
        
        # 将提示标签放在画布中央
        canvas_layout = QVBoxLayout(self.canvas_widget)
        canvas_layout.addStretch()
        canvas_layout.addWidget(self.hint_label)
        canvas_layout.addStretch()
        
        # 仅检测页面有编辑线段按钮
        if "检测" in self.title:
            preview_layout.addWidget(preview_title)
            
            # 直接检测页面
            preview_layout.addWidget(self.canvas_widget)
            # 使用水平布局来右对齐按钮
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            # 将编辑线段按钮保存为页面属性
            self.edit_button = QPushButton("编辑线段")
            self.edit_button.clicked.connect(self.open_line_edit_dialog)
            self.edit_button.setStyleSheet("""
                QPushButton {
                    background-color: #4A7A8C;
                    color: white;
                    padding: 6px 12px;
                    min-height: 28px;
                }
                QPushButton:hover {
                    background-color: #5A8AA0;
                }
                QPushButton:pressed {
                    background-color: #3A6A7C;
                }
            """)
            button_layout.addWidget(self.edit_button)
            preview_layout.addLayout(button_layout)
        else:
            # 非检测页面
            preview_layout.addWidget(preview_title)
            preview_layout.addWidget(self.canvas_widget)
        
        # 添加到主布局
        main_layout.addWidget(control_panel)
        main_layout.addWidget(preview_panel, 1)
    
    def select_pcd_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择PCD文件", "", "PCD Files (*.pcd)")
        if file_path:
            self.pcd_file_button.setText(os.path.basename(file_path))
    
    def select_config_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "", "YAML Files (*.yaml)")
        if file_path:
            self.config_file_button.setText(os.path.basename(file_path))
    
    def open_line_edit_dialog(self):
        print("这是lidar_gui创建的界面")

        # 获取当前脚本所在目录
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # # 构建line_editor_qt.py的完整路径
        # editor_script = os.path.join(script_dir, 'line_editor_qt.py')
        # # 使用subprocess启动line_editor_qt脚本
        # subprocess.Popen([sys.executable, editor_script])
    
    def reset_parameters(self):
        """
        重置当前页面的所有参数和文件选择
        """
        try:
            # 重置PCD文件选择
            if hasattr(self, 'pcd_file_button'):
                self.pcd_file_button.setText("请选择PCD文件")
            
            # 重置配置文件选择（如果存在）
            if hasattr(self, 'config_file_button'):
                self.config_file_button.setText("请选择yaml文件")
            
            # 重置参数设置区域的所有参数
            # 点云预览页面
            if self.title == "点云预览" and hasattr(self, 'param_spin_boxes_preview') and self.param_spin_boxes_preview:
                for i, spin_box in enumerate(self.param_spin_boxes_preview):
                    if i < len(self.param_defaults_preview):
                        try:
                            spin_box.setValue(self.param_defaults_preview[i][1])
                        except RuntimeError:
                            print("警告: 无法设置预览参数值，控件可能已被删除")
            # 点云投影页面
            elif self.title == "点云投影" and hasattr(self, 'param_spin_boxes_projection') and self.param_spin_boxes_projection:
                for i, spin_box in enumerate(self.param_spin_boxes_projection):
                    if i < len(self.param_defaults_projection):
                        try:
                            spin_box.setValue(self.param_defaults_projection[i][1])
                        except RuntimeError:
                            print("警告: 无法设置投影参数值，控件可能已被删除")
            # 直接检测页面
            elif self.title == "直接检测" and hasattr(self, 'param_spin_boxes_detect') and self.param_spin_boxes_detect:
                for i, spin_box in enumerate(self.param_spin_boxes_detect):
                    if i < len(self.param_defaults_detect):
                        try:
                            spin_box.setValue(self.param_defaults_detect[i][1])
                        except RuntimeError:
                            print("警告: 无法设置检测参数值，控件可能已被删除")
            
            # 重置提示文本 - 安全检查
            if "预览" in self.title:
                default_text = "请选择PCD文件并点击生成预览"
            elif "投影" in self.title:
                default_text = "请选择PCD文件并点击生成投影"
            elif "直接检测" in self.title:
                default_text = "请选择PCD文件并点击直接检测"
            
            # 检查hint_label是否存在且有效
            if hasattr(self, 'hint_label'):
                try:
                    # 尝试设置文本，如果控件已被删除会抛出RuntimeError
                    self.hint_label.setText(default_text)
                except RuntimeError:
                    # 如果hint_label已被删除，尝试重新创建它
                    print("检测到hint_label已被删除，重新创建")
                    # 首先检查canvas_widget是否存在
                    if hasattr(self, 'canvas_widget'):
                        # 获取canvas_widget的布局
                        layout = self.canvas_widget.layout()
                        if layout:
                            # 清空布局中的所有项
                            while layout.count() > 0:
                                item = layout.takeAt(0)
                                if item.widget():
                                    item.widget().deleteLater()
                            
                            # 重新创建hint_label
                            self.hint_label = QLabel(default_text)
                            self.hint_label.setAlignment(Qt.AlignCenter)
                            self.hint_label.setStyleSheet("color: #AAAAAA; font-size: 14px; padding: 10px;")
                            self.hint_label.setWordWrap(True)
                            
                            # 将新的hint_label添加到布局中
                            layout.addStretch()
                            layout.addWidget(self.hint_label)
                            layout.addStretch()
            
            # 如果是检测页面，清空检测结果
            if "检测" in self.title:
                for child in self.children():
                    if isinstance(child, QGroupBox) and child.title() == "线段信息":
                        for grand_child in child.children():
                            if isinstance(grand_child, QTextEdit):
                                try:
                                    grand_child.setText("")
                                    grand_child.setPlaceholderText("等待检测结果...")
                                except RuntimeError:
                                    print("警告: 无法更新线段信息文本控件")
                                break
        except Exception as e:
            print(f"重置参数时出错: {str(e)}")

# 导入QTextEdit
from PyQt5.QtWidgets import QTextEdit

class LidarVisualizer(QMainWindow):
    """LiDAR点云可视化主窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiDAR Point Cloud Studio v1.0")
        self.setGeometry(100, 100, 1440, 1024)
        self.initUI()
        self.setStyleSheet(self.get_dark_stylesheet())
    
    def initUI(self):
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)  # 导航栏与主内容区域无间隙
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 左侧导航栏
        nav_panel = QWidget()
        nav_panel.setMaximumWidth(140)  # 导航栏宽度保持不变
        nav_panel.setMinimumWidth(140)
        nav_panel.setStyleSheet("background-color: #2A2A2A; border-right: 1px solid #555555;")
        nav_layout = QVBoxLayout(nav_panel)
        nav_layout.setSpacing(5)
        nav_layout.setContentsMargins(8, 15, 8, 15)
        
        # 导航按钮
        self.nav_buttons = []
        nav_items = ["点云预览", "点云投影", "直接检测"]
        
        for i, item in enumerate(nav_items):
            button = QPushButton(item)
            button.setMinimumHeight(60)
            button.setCheckable(True)
            button.clicked.connect(lambda checked, idx=i: self.switch_page(idx))
            # 设置按钮样式，确保文本清晰显示
            button.setStyleSheet("""
                QPushButton {
                    background-color: #555555;
                    color: #CCCCCC;
                    font-size: 14px;
                    font-family: 'SimHei', 'Microsoft YaHei', sans-serif;
                    border: 1px solid #666666;
                }
                QPushButton:hover {
                    background-color: #666666;
                }
            """)
            self.nav_buttons.append(button)
            nav_layout.addWidget(button)
            nav_layout.addSpacing(5)
        
        nav_layout.addStretch()
        
        # 右侧主内容区域
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        
        # 创建各个页面
        self.pages = [
            MainPage("点云预览"),
            MainPage("点云投影"),
            MainPage("直接检测")
        ]
        
        # 添加第一个页面
        self.content_layout.addWidget(self.pages[0])
        
        # 激活第一个按钮
        self.nav_buttons[0].setChecked(True)
        self.highlight_button(0)
        
        # 添加到主布局
        main_layout.addWidget(nav_panel)
        main_layout.addWidget(self.content_widget, 1)
    
    def switch_page(self, index):
        # 清除当前内容
        while self.content_layout.count() > 0:
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.hide()
                self.content_layout.removeWidget(widget)
        
        # 添加新页面
        self.content_layout.addWidget(self.pages[index])
        self.pages[index].show()
        
        # 高亮当前按钮
        self.highlight_button(index)
    
    def highlight_button(self, index):
        for i, button in enumerate(self.nav_buttons):
            if i == index:
                button.setStyleSheet("""
                    background-color: #4A7A8C;
                    color: white;
                    font-size: 14px;
                    font-family: 'SimHei', 'Microsoft YaHei', sans-serif;
                    border: 1px solid #5A8AA0;
                    font-weight: bold;
                """)
            else:
                button.setStyleSheet("""
                    background-color: #555555;
                    color: #CCCCCC;
                    font-size: 14px;
                    font-family: 'SimHei', 'Microsoft YaHei', sans-serif;
                    border: 1px solid #666666;
                """)
    
    def get_dark_stylesheet(self):
        return """
            QMainWindow, QWidget {
                background-color: #333333;
                color: #FFFFFF;
                font-size: 14px;
                font-family: 'SimHei', 'Microsoft YaHei', sans-serif;
            }
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 4px;
                margin-top: 10px;
                margin-bottom: 10px;
                padding: 20px 10px 10px 10px;
                background-color: #2A2A2A;
            }
            QGroupBox::title {
                subcontrol-origin: padding;
                subcontrol-position: top left; /* 标题放在内部左上角 */
                left: 10px;
                top: 2px; /* 标题位置 */
                padding: 0 5px;
                color: #FFFFFF;
                font-size: 14px;
                background-color: transparent; /* 透明背景 */
                border: none; /* 无边框 */
            }
            QLineEdit {
                background-color: #444444;
                border: 1px solid #666666;
                padding: 3px;
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #555555;
                border: 1px solid #666666;
                padding: 8px 12px;
                color: #CCCCCC;
                font-size: 14px;
                font-family: 'SimHei', 'Microsoft YaHei', sans-serif;
                min-height: 32px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QPushButton:pressed {
                background-color: #333333;
                border-color: #444444;
            }
            QPushButton:checked {
                background-color: #4A7A8C;
                color: white;
            }
            QLabel {
                color: #FFFFFF;
            }
            QTextEdit {
                background-color: #444444;
                border: 1px solid #666666;
                color: #FFFFFF;
                padding: 5px;
            }
        """

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置应用程序样式
    app.setStyle("Fusion")
    
    # 调整全局字体 - 增大字体大小以确保文本清晰显示
    font = QFont("SimHei", 12)
    app.setFont(font)
    
    # 创建并显示主窗口
    window = LidarVisualizer()
    window.show()
    
    sys.exit(app.exec_())