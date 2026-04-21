# -*- coding: utf-8 -*-
# 简易本地API：上传PCD -> 返回三类结果（预览/一次检测/多阶段）
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn, os, shutil, time

# === 引用你已封装的函数 ===
from preprocess.datasets.pcd_preview import preview_pcd_screenshot        # 预览（离屏png）
from preprocess.datasets.pcd2line_test import run_pcd2line_once           # 一次检测
from preprocess.datasets.pcd2lines_step import run_pcd2lines_multistage   # 多阶段检测

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
    message: str
    preview_png: str | None = None
    once_png: str | None = None
    multi_pngs: dict | None = None   # {"stage1_long":path, "stage2_short":path, "final_merge":path}
    line_counts: dict | None = None  # {"once": int, "long": int, "short": int, "final": int}
    out_dir: str

def _save_upload(file: UploadFile) -> str:
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f: shutil.copyfileobj(file.file, f)
    return path

@app.post("/preview", response_model=DetectResult)
async def preview_endpoint(file: UploadFile = File(...)):
    pcd_path = _save_upload(file)
    # 生成预览
    png_path = os.path.join(OUT_DIR, "preview", "preview.png")
    ret = preview_pcd_screenshot(pcd_path, png_path, width=1600, height=1200, point_size=2.5, bg_white=True)
    return DetectResult(message="ok", preview_png=ret["png"], out_dir=OUT_DIR)

@app.post("/detect/once", response_model=DetectResult)
async def detect_once_endpoint(
    file: UploadFile = File(...),
    cfg: str = Form(CFG_DEFAULT),
):
    pcd_path = _save_upload(file)
    # 一次检测
    out_dir = os.path.join(OUT_DIR, "once")
    os.makedirs(out_dir, exist_ok=True)
    res = run_pcd2line_once(pcd_path, cfg_path=cfg, save_dir=out_dir, show=False)
    once_png = os.path.join(out_dir, "once_detect.png")
    return DetectResult(
        message="ok",
        once_png=once_png,
        line_counts={"once": len(res["linesegs"].linesegments)},
        out_dir=out_dir
    )

@app.post("/detect/multi", response_model=DetectResult)
async def detect_multi_endpoint(
    file: UploadFile = File(...),
    cfg: str = Form(CFG_DEFAULT),
):
    pcd_path = _save_upload(file)
    # 多阶段检测
    out_dir = os.path.join(OUT_DIR, "multi")
    os.makedirs(out_dir, exist_ok=True)
    res = run_pcd2lines_multistage(pcd_path, cfg_path=cfg, save_dir=out_dir, show=False)
    multi_pngs = {
        "stage1_long": os.path.join(out_dir, "stage1_long.png"),
        "stage2_short": os.path.join(out_dir, "stage2_short.png"),
        "final_merge":  os.path.join(out_dir, "final_merge.png"),
    }
    counts = {
        "long":  len(res["lines_long"].linesegments),
        "short": len(res["lines_short"].linesegments),
        "final": len(res["lines_all"].linesegments),
    }
    return DetectResult(message="ok", multi_pngs=multi_pngs, line_counts=counts, out_dir=out_dir)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
