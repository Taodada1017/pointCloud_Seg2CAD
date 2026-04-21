# -*- coding: utf-8 -*-
"""
GUI API适配器使用示例
演示如何在现有的LidarVisualizer应用程序中集成gui_api_adapter
"""

# 这是一个示例文件，展示如何在您现有的GUI应用程序中使用我们的适配器
# 请根据您的实际代码结构进行相应的修改

# 1. 导入必要的模块
import sys
from PyQt5.QtWidgets import QApplication

# 2. 导入您的GUI类
from lidar_gui import LidarVisualizer

# 3. 导入我们的适配器
from gui_api_adapter import connect_gui_with_api


def main():
    """主函数 - 演示如何集成适配器"""
    # 创建应用程序实例
    app = QApplication(sys.argv)
    
    # 创建LidarVisualizer主窗口实例
    main_window = LidarVisualizer()
    
    # 这是关键步骤：使用适配器连接GUI和API
    # 不需要修改lidar_gui.py中的任何代码
    # 适配器会自动连接所有按钮和处理事件
    adapter = connect_gui_with_api(main_window)
    
    # 显示主窗口
    main_window.show()
    
    # 启动应用程序事件循环
    sys.exit(app.exec_())


if __name__ == "__main__":
    # 启动应用程序
    main()