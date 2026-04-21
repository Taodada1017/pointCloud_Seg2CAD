import sys
import os
import numpy as np
from PyQt5.QtWidgets import QApplication

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 现在可以正确导入模块了
from preprocess.geometry.lineseg import LineSegment, LineSegments
from preprocess.datasets.line_editor_qt import LineEditorComponent, LineEditorCanvas

# 创建一个简单的测试应用
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 创建LineEditorComponent实例
    editor = LineEditorComponent()
    
    # 创建一个简单的图像和线段进行测试
    # 创建一个500x500的黑色图像
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # 创建一些测试线段
    lines = []
    # 添加几条测试线段，使用numpy数组而不是元组
    lines.append(LineSegment(np.array([100, 100]), np.array([400, 100])))
    lines.append(LineSegment(np.array([400, 100]), np.array([400, 400])))
    lines.append(LineSegment(np.array([400, 400]), np.array([100, 400])))
    lines.append(LineSegment(np.array([100, 400]), np.array([100, 100])))
    lines.append(LineSegment(np.array([100, 100]), np.array([400, 400])))
    
    linesegs = LineSegments(lines)
    
    # 使用from_data_source方法设置数据
    editor.from_data_source(img=img, linesegs=linesegs)
    
    # 测试保存DXF文件
    dxf_path = "test_output.dxf"
    result_dxf = editor.save_dxf(dxf_path)
    print(f"保存DXF文件 {'成功' if result_dxf else '失败'}: {dxf_path}")
    
    # 测试保存图像
    img_path = "test_output.png"
    result_img = editor.save_image(img_path)
    print(f"保存图像文件 {'成功' if result_img else '失败'}: {img_path}")
    
    print("测试完成！请检查生成的文件。")
    
    # 显示测试结果
    if result_dxf and result_img:
        print("重构后的line_editor_qt.py功能正常！")
    else:
        print("警告：部分功能测试失败，请检查错误信息。")
    
    # 如果需要可以显示界面进行交互测试
    # editor.show()
    # sys.exit(app.exec_())