# -*- coding: utf-8 -*-
"""
测试GUI与API连接的脚本
用于启动GUI并验证与API的连接是否正常工作
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication, QMessageBox

# 尝试导入LidarVisualizer
try:
    from lidar_gui import LidarVisualizer
    gui_imported = True
except ImportError as e:
    print(f"警告: 无法导入lidar_gui模块 - {e}")
    gui_imported = False

# 导入我们的适配器
from gui_api_adapter import connect_gui_with_api

def main():
    """主函数"""
    # 检查是否导入了GUI模块
    if not gui_imported:
        # 如果无法导入，尝试使用备选方案
        print("尝试使用备选GUI导入方式...")
        return run_with_alternate_gui()
    
    print("成功导入LidarVisualizer类")
    
    # 创建应用程序实例
    app = QApplication(sys.argv)
    
    # 设置应用程序样式和字体
    app.setStyle("Fusion")
    
    # 创建主窗口
    main_window = LidarVisualizer()
    
    # 使用我们的适配器连接GUI和API
    adapter = connect_gui_with_api(main_window)
    
    # 显示主窗口
    main_window.show()
    
    # 启动应用程序事件循环
    sys.exit(app.exec_())

def run_with_alternate_gui():
    """备选GUI启动方案
    如果无法直接导入LidarVisualizer，则尝试创建一个简单的GUI界面来测试API连接"""
    # 创建应用程序实例
    app = QApplication(sys.argv)
    
    # 显示一个对话框说明情况
    reply = QMessageBox.information(
        None,
        "GUI测试",
        "无法直接导入LidarVisualizer类。\n\n" \
        "您可以：\n" \
        "1. 在您现有的GUI文件中添加以下代码来使用适配器：\n\n" \
        "from gui_api_adapter import connect_gui_with_api\n" \
        "# 在创建LidarVisualizer实例后\n" \
        "main_window = LidarVisualizer()\n" \
        "adapter = connect_gui_with_api(main_window)\n" \
        "main_window.show()\n\n2. 或者修改gui_api_adapter.py以适应您的具体GUI结构",
        QMessageBox.Ok
    )
    
    # 直接退出
    return

if __name__ == "__main__":
    # 确保API服务已启动
    try:
        from api_client import api_client
        if not api_client.is_api_running():
            QMessageBox.warning(
                None,
                "API服务未运行",
                "API服务似乎未运行。\n" \
                "请先在另一个终端运行 'python api.py' 来启动API服务，\n" \
                "然后再运行此测试脚本。"
            )
    except ImportError:
        print("警告: 无法导入api_client模块")
    
    # 运行主函数
    main()