"""
内存优化版 main.py - 分阶段加载模型，避免 OOM
结果与原版完全一致，只是模型不同时驻留内存
"""
import argparse
import os
import sys, pathlib
import numpy as np
import json
import torch
import gc

# ---- 路径设置 ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "GroundingDINO"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "segment_anything"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "recognize-anything"))

from PIL import Image
import cv2
import torchvision
import torchvision.transforms as TS
import matplotlib
matplotlib.use('Agg')  # 无头模式，减少内存
import matplotlib.pyplot as plt
import time

# GroundingDINO
from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util.slconfig import SLConfig
import groundingdino.datasets.transforms as T

# SAM
from segment_anything import sam_model_registry, SamPredictor

# RAM
from ram.models import ram
from ram import inference_ram

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    logits.shape[0]
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())
    return boxes_filt, torch.Tensor(scores), pred_phrases

from groundingdino.util.utils import get_phrases_from_posmap

def save_mask_data(output_dir, image_name, raw_tags, masks, boxes, labels):
    value = 0
    mask_img = torch.zeros(masks.shape[-2:])
    mask_list = []
    mask_list.append({"value": value, "label": "background"})
    for idx, mask in enumerate(masks):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        label = labels[idx]
        word = label.split("(")[0]
        score = float(label.split("(")[1].replace(")", ""))
        words = word.split()
        labels_dict = {w: score for w in words}
        mask_list.append({
            "value": value + idx + 1,
            "labels": labels_dict,
            "box": boxes[idx].numpy().tolist()
        })
    mask_img_np = mask_img.numpy().astype(np.uint16)
    mask_path = os.path.join(output_dir, f"{image_name}_mask.png")
    cv2.imwrite(mask_path, mask_img_np)
    tags_text = raw_tags.replace(".", ". ").strip()
    if tags_text.endswith("."):
        tags_text = tags_text[:-1].strip()
    json_data = {
        "raw_tags": raw_tags.replace(".", " .").strip(),
        "tags": tags_text,
        "mask": mask_list
    }
    json_path = os.path.join(output_dir, f"{image_name}_label.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    config_file = os.path.join(SCRIPT_DIR, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    weights_dir = os.environ.get("WEIGHTS_DIR", os.path.join(SCRIPT_DIR, "weights"))
    ram_checkpoint = os.path.join(weights_dir, "ram_swin_large_14m.pth")
    grounded_checkpoint = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")
    sam_checkpoint = os.path.join(weights_dir, "sam_vit_h_4b8939.pth")

    input_folder = "assets"
    output_base_dir = "outputs"
    if len(sys.argv) == 3:
        input_folder = sys.argv[1]
        output_base_dir = sys.argv[2]
        print(f"输入路径: {input_folder}")
        print(f"输出路径: {output_base_dir}")

    sam_version = "vit_h"
    box_threshold = 0.3
    text_threshold = 0.25
    iou_threshold = 0.5
    device = "cpu"

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(supported_formats)])

    # 跳过已处理的图像
    existing = set()
    if os.path.exists(output_base_dir):
        for f in os.listdir(output_base_dir):
            if f.endswith("_mask.png"):
                existing.add(f.replace("_mask.png", ""))
    remaining = [f for f in image_files if os.path.splitext(f)[0] not in existing]
    print(f"总共 {len(image_files)} 张, 已完成 {len(existing)} 张, 剩余 {len(remaining)} 张")

    if not remaining:
        print("所有图像已处理完成！")
        exit(0)

    os.makedirs(output_base_dir, exist_ok=True)
    vis_output_dir = os.path.join(output_base_dir, "other")
    os.makedirs(vis_output_dir, exist_ok=True)

    # ===== 阶段 A: RAM 批量提取标签 =====
    print("\n===== 加载 RAM 模型 =====")
    t0 = time.time()
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = TS.Compose([TS.Resize((384, 384)), TS.ToTensor(), normalize])
    ram_model = ram(pretrained=ram_checkpoint, image_size=384, vit='swin_l')
    ram_model.eval()
    ram_model = ram_model.to(device)
    print(f"RAM 加载耗时: {time.time()-t0:.1f}s")

    tags_dict = {}  # image_name -> raw_tags
    for i, image_file in enumerate(remaining):
        image_path = os.path.join(input_folder, image_file)
        image_name = os.path.splitext(image_file)[0]
        image_pil = Image.open(image_path).convert("RGB")
        raw_image_ram = image_pil.resize((384, 384))
        raw_image_ram = transform(raw_image_ram).unsqueeze(0).to(device)
        with torch.no_grad():
            res = inference_ram(raw_image_ram, ram_model)
        raw_tags = res[0].replace(' |', '.')
        tags_dict[image_name] = raw_tags
        print(f"[RAM {i+1}/{len(remaining)}] {image_file}: {res[0][:80]}...")
        del raw_image_ram, image_pil
        free_memory()

    # 释放 RAM 模型
    del ram_model, transform, normalize
    free_memory()
    print(f"RAM 完成，已释放内存。标签数: {len(tags_dict)}")

    # ===== 阶段 B: GroundingDINO 检测 =====
    print("\n===== 加载 GroundingDINO 模型 =====")
    t0 = time.time()
    dino_model = load_model(config_file, grounded_checkpoint, device=device)
    print(f"DINO 加载耗时: {time.time()-t0:.1f}s")

    detections_dict = {}  # image_name -> (boxes_filt, scores, pred_phrases)
    for i, image_file in enumerate(remaining):
        image_path = os.path.join(input_folder, image_file)
        image_name = os.path.splitext(image_file)[0]
        _, image_tensor = load_image(image_path)
        raw_tags = tags_dict[image_name]
        with torch.no_grad():
            boxes_filt, scores, pred_phrases = get_grounding_output(
                dino_model, image_tensor, raw_tags, box_threshold, text_threshold, device=device
            )
        # NMS
        if boxes_filt.shape[0] > 0:
            image_pil = Image.open(image_path).convert("RGB")
            W, H = image_pil.size
            for j in range(boxes_filt.size(0)):
                boxes_filt[j] = boxes_filt[j] * torch.Tensor([W, H, W, H])
                boxes_filt[j][:2] -= boxes_filt[j][2:] / 2
                boxes_filt[j][2:] += boxes_filt[j][:2]
            boxes_filt = boxes_filt.cpu()
            nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
            boxes_filt = boxes_filt[nms_idx]
            pred_phrases = [pred_phrases[idx] for idx in nms_idx]
            scores = scores[nms_idx]
            del image_pil
        detections_dict[image_name] = (boxes_filt, scores, pred_phrases)
        print(f"[DINO {i+1}/{len(remaining)}] {image_file}: {boxes_filt.shape[0]} boxes")
        del image_tensor
        free_memory()

    del dino_model
    free_memory()
    print(f"DINO 完成，已释放内存。检测数: {len(detections_dict)}")

    # ===== 阶段 C: SAM 分割 + 保存 =====
    print("\n===== 加载 SAM 模型 =====")
    t0 = time.time()
    sam = sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)
    print(f"SAM 加载耗时: {time.time()-t0:.1f}s")

    for i, image_file in enumerate(remaining):
        t1 = time.time()
        image_path = os.path.join(input_folder, image_file)
        image_name = os.path.splitext(image_file)[0]
        raw_tags = tags_dict[image_name]
        boxes_filt, scores, pred_phrases = detections_dict[image_name]

        if boxes_filt.shape[0] == 0:
            print(f"[SAM {i+1}/{len(remaining)}] {image_file}: 无检测框，跳过")
            continue

        image_cv = cv2.imread(image_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_cv)

        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2]).to(device)
        with torch.no_grad():
            masks, _, _ = predictor.predict_torch(
                point_coords=None, point_labels=None,
                boxes=transformed_boxes.to(device), multimask_output=False
            )

        # 保存可视化
        plt.figure(figsize=(10, 10))
        plt.imshow(image_cv)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)
        plt.axis('off')
        plt.savefig(os.path.join(vis_output_dir, f"{image_name}_automatic_label_output.jpg"),
                    bbox_inches="tight", dpi=300, pad_inches=0.0)
        plt.close('all')

        # 保存 mask + label
        save_mask_data(output_base_dir, image_name, raw_tags, masks, boxes_filt, pred_phrases)

        del image_cv, masks, transformed_boxes
        free_memory()

        print(f"[SAM {i+1}/{len(remaining)}] {image_file}: {boxes_filt.shape[0]} masks, {time.time()-t1:.1f}s")

    del sam, predictor
    free_memory()

    total_masks = len([f for f in os.listdir(output_base_dir) if f.endswith("_mask.png")])
    print(f"\n===== 全部完成！共 {total_masks} 个 mask 文件 =====")
