# -*- coding: utf-8 -*-
# 简易本地API：上传PCD -> 返回三类结果（预览/一次检测/多阶段）
# 线段编辑API已分离到 line_editor_api.py 文件中，端口为8001
# 自动将项目根目录加入sys.path
import sys
import os
import numpy as np

current_file = os.path.abspath(__file__)
# 向上跳三级：datasets → preprocess → 根目录
root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
#print("修正后root_path:", root_path)  # 验证是否为项目根目录
sys.path.append(root_path)
    
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn, os, shutil, time
import traceback
from fastapi import HTTPException

# === 引用你已封装的函数 ===
from preprocess.datasets.pcd_preview import preview_pcd_screenshot        # 预览（离屏png）
from preprocess.datasets.pcd_preview_v2 import run_pcd_preview           # 预览v2（可配置视角）
from preprocess.datasets.pcd2line_test import run_pcd2line_once           # 一次检测
from preprocess.datasets.pcd2lines_step import run_pcd2lines_multistage   # 多阶段检测
from preprocess.datasets.pcd_projection import run_pcd_projection         # 3D点云转2D投影

from preprocess.geometry.lineseg import LineSegment, LineSegments

# === 基础路径 ===
ROOT = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(ROOT, "uploads")
OUT_DIR    = os.path.join(ROOT, "output")
CFG_DEFAULT = r"D:\work\l2bim\configs\interval\15m\1F\1f_office_03.yaml"  # 你的cfg默认值

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

app = FastAPI(title="LiDAR2BIM Demo API", version="1.0")

# 允许本地页面联调
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# === 统一的返回体 ===
class DetectResult(BaseModel):
    """检测结果模型"""
    message: str
    preview_png: str | None = None
    once_png: str | None = None
    multi_pngs: dict | None = None   # {"stage1_long":path, "stage2_short":path, "final_merge":path}
    line_counts: dict | None = None  # {"once": int, "long": int, "short": int, "final": int}
    out_dir: str
    img: list | None = None  # 修改为list类型，因为JSON无法直接序列化numpy数组
    serialized_linesegs: list | None = None  # 直接使用序列化后的线段列表
    
    def dict(self, *args, **kwargs):
        """自定义序列化方法，确保img能正确序列化"""
        result = super().dict(*args, **kwargs)
        
        # 确保img是列表类型（已在赋值时处理）
        if 'img' in result and result['img'] is not None:
            print(f"序列化img: {type(result['img'])}")
        
        return result
 
def _save_upload(file: UploadFile) -> str:
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f: shutil.copyfileobj(file.file, f)
    return path


@app.post("/detect/once", response_model=DetectResult)
async def detect_once_endpoint(
    file: UploadFile = File(...),
    #cfg: str = Form(CFG_DEFAULT),
    cfg: str = Form(...),  # 移除默认值，设为必需参数
    return_numpy: bool = Form(False),  # 添加参数控制是否返回numpy数组
    return_linesegs: bool = Form(False),  # 添加参数控制是否返回线段数据
    min_line_length: int = Form(30),  # 添加参数控制最小线段长度
):
    pcd_path = _save_upload(file)
    # 一次检测
    out_dir = os.path.join(OUT_DIR, "once")
    os.makedirs(out_dir, exist_ok=True)
    try:
        res = run_pcd2line_once(
            pcd_path,
            cfg_path=cfg,
            save_dir=out_dir,
            show=False,
            min_line_length=min_line_length,
        )
    except Exception as e:
        print("[api] /detect/once 失败：", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    once_png = os.path.join(out_dir, "once_detect.png")
    
    # 创建响应对象
    detect_result = DetectResult(
        message="ok",
        once_png=once_png,
        line_counts={"once": len(res["linesegs"].linesegments)},
        out_dir=out_dir,
        img=None,
        serialized_linesegs=None

    )
    
    # 如果需要返回numpy数组，直接使用res中的img
    if return_numpy and 'img' in res and res['img'] is not None:
        try:
            # 在赋值前先将numpy数组转换为列表
            detect_result.img = res['img'].tolist()
            print(f"成功获取numpy数组图像并转换为列表: shape={res['img'].shape}, dtype={res['img'].dtype}")
        except Exception as e:
            print(f"无法获取或转换numpy数组图像: {str(e)}")
    
    # 处理linesegs，直接在这里进行序列化
    if return_linesegs and 'linesegs' in res and res['linesegs'] is not None:
        try:
            # 序列化LineSegments对象
            serialized_linesegs = []
            linesegs_obj = res['linesegs']
            print(f"开始序列化linesegs，原始对象包含线段数: {len(linesegs_obj.linesegments)}")
            
            # 尝试不同的方式获取线段数据
            # 1. 检查是否有linesegments属性
            if hasattr(linesegs_obj, 'linesegments'):
                for seg in linesegs_obj.linesegments:
                    try:
                        # 尝试多种可能的端点访问方式
                        point_a = []
                        point_b = []
                        
                        if hasattr(seg, 'A'):
                            point_a = seg.A.tolist() if hasattr(seg.A, 'tolist') else list(seg.A)
                        elif hasattr(seg, 'point_a'):
                            point_a = list(seg.point_a)
                        elif hasattr(seg, 'a'):
                            point_a = list(seg.a)
                        elif len(seg) >= 2:
                            point_a = list(seg[0])
                        
                        if hasattr(seg, 'B'):
                            point_b = seg.B.tolist() if hasattr(seg.B, 'tolist') else list(seg.B)
                        elif hasattr(seg, 'point_b'):
                            point_b = list(seg.point_b)
                        elif hasattr(seg, 'b'):
                            point_b = list(seg.b)
                        elif len(seg) >= 2:
                            point_b = list(seg[1])
                        
                        if point_a and point_b:
                            serialized_linesegs.append({
                                'point_a': point_a,
                                'point_b': point_b
                            })
                    except Exception as inner_e:
                        print(f"处理单条线段时出错: {str(inner_e)}")
                        continue
            # 2. 如果自身是可迭代对象，直接尝试迭代
            elif hasattr(linesegs_obj, '__iter__') and not isinstance(linesegs_obj, (str, dict)):
                for seg in linesegs_obj:
                    try:
                        if len(seg) >= 2:
                            serialized_linesegs.append({
                                'point_a': list(seg[0]) if hasattr(seg[0], '__iter__') else [0, 0, 0],
                                'point_b': list(seg[1]) if hasattr(seg[1], '__iter__') else [0, 0, 0]
                            })
                    except Exception as inner_e:
                        print(f"处理单条线段时出错: {str(inner_e)}")
                        continue
            
            detect_result.serialized_linesegs = serialized_linesegs
            print(f"成功序列化并设置linesegs字段，共{len(serialized_linesegs)}条线段")
        except Exception as e:
            print(f"序列化linesegs时出错: {str(e)}")
            detect_result.serialized_linesegs = []
    
    return detect_result

# @app.post("/detect/multi", response_model=DetectResult)
# async def detect_multi_endpoint(
#     file: UploadFile = File(...),
#     #cfg: str = Form(CFG_DEFAULT),
#     cfg: str = Form(...),  # 移除默认值，设为必需参数
# ):
#     pcd_path = _save_upload(file)
#     # 多阶段检测
#     out_dir = os.path.join(OUT_DIR, "multi")
#     os.makedirs(out_dir, exist_ok=True)
#     res = run_pcd2lines_multistage(pcd_path, cfg_path=cfg, save_dir=out_dir, show=False)
#     multi_pngs = {
#         "stage1_long": os.path.join(out_dir, "stage1_long.png"),
#         "stage2_short": os.path.join(out_dir, "stage2_short.png"),
#         "final_merge":  os.path.join(out_dir, "final_merge.png"),
#     }
#     counts = {
#         "long":  len(res["lines_long"].linesegments),
#         "short": len(res["lines_short"].linesegments),
#         "final": len(res["lines_all"].linesegments),
#     }
#     return DetectResult(message="ok", multi_pngs=multi_pngs, line_counts=counts, out_dir=out_dir)

@app.post("/projection", response_model=DetectResult)
async def projection_endpoint(
    file: UploadFile = File(...),
    cfg: str = Form(...),  # 移除默认值，设为必需参数
    save_png: bool = Form(True),
    projection_scale: float = Form(...),
):
    pcd_path = _save_upload(file)
    # 3D点云转2D投影
    out_dir = os.path.join(OUT_DIR, "projection")
    os.makedirs(out_dir, exist_ok=True)
    try:
        res = run_pcd_projection(
            pcd_path=pcd_path,
            cfg_path=cfg,
            save_dir=out_dir,
            save_png=save_png,
            projection_scale=projection_scale,
        )
    except Exception as e:
        print("[api] /projection 失败：", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    return DetectResult(
        message="ok",
        preview_png=res["png"],
        line_counts={"num_points": res["num_points"]},
        out_dir=out_dir
    )

@app.post("/preview/v2", response_model=DetectResult)
async def preview_v2_endpoint(
    file: UploadFile = File(...),
    theta: float = Form(0.0),
    phi: float = Form(0.0),
    distance: float = Form(2.0),
    background_color: float = Form(0.0),
    voxel_size: float = Form(0.01),
):
    pcd_path = _save_upload(file)
    # 点云预览v2（可配置视角）
    out_dir = os.path.join(OUT_DIR, "preview")
    os.makedirs(out_dir, exist_ok=True)
    
    # 生成唯一的输出文件名，避免覆盖
    timestamp = int(time.time())
    output_filename = f"preview_v2_{timestamp}.png"
    
    res = run_pcd_preview(
        pcd_path=pcd_path,
        theta=theta,
        phi=phi,
        distance=distance,
        background_color=background_color,
        voxel_size=voxel_size,
        save_dir=out_dir,
        output_filename=output_filename
    )
    
    return DetectResult(
        message="ok",
        preview_png=res["preview_path"],
        line_counts={
            "num_points_original": res["num_points_original"],
            "num_points_downsampled": res["num_points_downsampled"]
        },
        out_dir=out_dir
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
