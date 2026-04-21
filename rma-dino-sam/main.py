import argparse
import os
import sys, pathlib
import numpy as np
import json
import torch
import torchvision
from PIL import Image
#import litellm

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment_anything
from segment_anything.segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model
from ram.models import ram
from ram import inference_ram
import torchvision.transforms as TS

import time

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # 转化为RGB

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333), #调整尺寸
            T.ToTensor(),  # 转为张量
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #标准化
        ]
    )
    image, _ = transform(image_pil, None)  #
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)     # 加载配置文件
    args.device = device
    model = build_model(args)       # 构建模型结构
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")  # 加载权重
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cpu"):
    caption = caption.lower()   # 文本预处理
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption]) # 逻辑推理

    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)预测置信度
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)预测边界框
    logits.shape[0]

    # 过滤低置信度
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # 提取文本短语
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases

# 掩码
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# 展示检测框
def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def get_after_last_space(s):
    if ' ' in s:
        return s.rsplit(' ', 1)[-1]
    return s


# 保存掩码数据 mask.jpg 和 label.json作为FM-Fusion的输入。 后续考虑把主程序中输出文件去掉或者不保存在一起。
def save_mask_data(output_dir, image_name,raw_tags, mask_list, box_list, label_list):
    value = 0  # 0 for background
    # print(label_list)
    mask_img = torch.zeros(mask_list.shape[-2:])

    for idx, mask in enumerate(mask_list):
        #print("现在的idx: ",idx)
        mask_img[mask.cpu().numpy()[0] == True] = idx + 1

    # 保存为16位PNG（保留原始数值）
    mask_array = mask_img.numpy().astype(np.uint16)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_mask.png"), mask_array)

    # 可视化保存mask
    # plt.figure(figsize=(10, 10))
    # plt.imshow(mask_img.numpy())          # 将数值转换为颜色
    # plt.axis('off')
    # plt.savefig(os.path.join(output_dir, f"{image_name}_mask.jpg"), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = {
        'raw_tags': raw_tags,
        'tags': "",
        'mask':[{
            'value': value,
            'label': 'background'
        }]
    }
    tag_set = set()
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')

        # name = get_after_last_space(name)
        # tag_set.add(name)
        # logit = logit[:-1] # the last is ')'
        # label_json = {}
        # label_json[name] = float(logit)
        logit = logit[:-1]
        label_json = {}

        words = name.split(' ')
        for word in words:
            tag_set.add(word)
            label_json[word] = float(logit)

        json_data['mask'].append({
            'value': value,
            'labels': label_json,
            'box': box.numpy().tolist(),
        })
    tags = ""
    for tag in tag_set:
        tags += tag + ". "
    tags = tags[:-2]
    json_data['tags'] = tags

    with open(os.path.join(output_dir, f"{image_name}_label.json"), 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":

    start = time.time()
    # by Cancer

    # 本地配置
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # GroundingDINO配置文件

    # 模型权重路径：优先使用环境变量 WEIGHTS_DIR，否则使用默认路径
    # 在服务器上设置: export WEIGHTS_DIR="/path/to/your/weights"
    weights_dir = os.environ.get("WEIGHTS_DIR", os.path.join(os.path.dirname(__file__), "weights"))
    ram_checkpoint = os.path.join(weights_dir, "ram_swin_large_14m.pth")  # RAM模型路径
    grounded_checkpoint = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")  # DINO模型路径
    sam_checkpoint = os.path.join(weights_dir, "sam_vit_h_4b8939.pth")  # sam模型路径

    # 修改为输入输出文件夹路径
    input_folder = "assets"  # 输入图像的文件夹路径
    output_base_dir = "outputs"  # 基础输出文件夹路径

    if (len(sys.argv) == 3):
        input_folder = sys.argv[1]
        output_base_dir = sys.argv[2]
        print("指令读取成功")
        print("输入路径", input_folder)
        print("输出路径", output_base_dir)

    else:
        print("指令读取失败，输入输出采用默认路径")
        print("输入路径", input_folder)
        print("输出路径", output_base_dir)

    # 以下配置无需修改
    sam_version = "vit_h"  # sam模型类型
    sam_hq_checkpoint = None
    use_sam_hq = None  # 这两行调为None不管， 因为没用sam_hq模式
    split = ","  # 标签分隔符
    box_threshold = 0.3  # 这三行参考源码的default
    text_threshold = 0.25
    iou_threshold = 0.5   # NMS 的重叠阈值， 超过则判断为同一物体，防止重复检测
    device = "cpu"  # 使用的是cpu版本

    # 获取输入文件夹中的所有图像文件
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(supported_formats)]

    if not image_files:
        print(f"在文件夹 {input_folder} 中没有找到图像文件")
        exit(1)

    print(f"找到 {len(image_files)} 个图像文件: {image_files}")

    # 加载模型（放在循环外面，避免重复加载）
    print("正在加载模型...")

    # 加载GroundingDINO模型
    model = load_model(config_file, grounded_checkpoint, device=device)

    # 加载RAM模型
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
        TS.Resize((384, 384)),
        TS.ToTensor(), normalize
    ])

    ram_model = ram(pretrained=ram_checkpoint, image_size=384, vit='swin_l')
    ram_model.eval()
    ram_model = ram_model.to(device)

    # 加载SAM模型
    predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

    end = time.time()
    print(f"加载模型耗时: {end - start:.2f} 秒，开始处理图像...")

    os.makedirs(output_base_dir, exist_ok=True)
    vis_output_dir = os.path.join(output_base_dir, "other")
    os.makedirs(vis_output_dir, exist_ok=True)

    # 循环处理每个图像文件
    for image_file in image_files:
        start = time.time()
        try:
            # 构建完整的图像路径
            image_path = os.path.join(input_folder, image_file)

            # 获取图像文件名（不含扩展名）用于输出前缀
            image_name = os.path.splitext(image_file)[0]

            output_dir = output_base_dir
            #os.makedirs(output_dir, exist_ok=True)

            print(f"\n正在处理图像: {image_file}")
            print(f"输出目录: {output_dir}")

            # 加载当前图像
            image_pil, image = load_image(image_path)

            # 保存原始图像（添加前缀）
            # raw_image_path = os.path.join(output_dir, f"{image_name}_raw_image.jpg")
            # image_pil.save(raw_image_path)

            # 使用RAM模型生成标签
            start1 = time.time()

            raw_image_ram = image_pil.resize((384, 384))
            raw_image_ram = transform(raw_image_ram).unsqueeze(0).to(device)

            res = inference_ram(raw_image_ram, ram_model)
            raw_tags = res[0].replace(' |', '.')
            tags_chinese = res[1].replace(' |', ',')

            print(type(raw_tags))

            end1 = time.time()
            print(f"ram: {end1 - start1:.2f} 秒")

            print(f"图像标签 (英文): {res[0]}")
            print(f"图像标签 (中文): {res[1]}")

            # 运行GroundingDINO模型进行目标检测
            start2 = time.time()

            boxes_filt, scores, pred_phrases = get_grounding_output(
                model, image, raw_tags, box_threshold, text_threshold, device=device
            )

            end2 = time.time()
            print(f"DINO: {end2 - start2:.2f} 秒")

            # 设置SAM预测器的图像
            start3 = time.time()

            image_cv = cv2.imread(image_path)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_cv)

            # 转换边界框坐标
            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()

            # 使用NMS处理重叠框
            print(f"Before NMS: {boxes_filt.shape[0]} boxes")
            nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
            boxes_filt = boxes_filt[nms_idx]
            pred_phrases = [pred_phrases[idx] for idx in nms_idx]
            print(f"After NMS: {boxes_filt.shape[0]} boxes")

            # 如果检测到目标，则进行分割
            if boxes_filt.shape[0] > 0:

                transformed_boxes = predictor.transform.apply_boxes_torch(
                    boxes_filt,
                    image_cv.shape[:2]
                ).to(device)

                # 生成掩码
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes.to(device),
                    multimask_output=False,
                )

                end3 = time.time()
                print(f"sam: {end3 - start3:.2f} 秒")

                # 绘制图像
                plt.figure(figsize=(10, 10))
                plt.imshow(image_cv)
                for mask in masks:
                    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                for box, label in zip(boxes_filt, pred_phrases):
                    show_box(box.numpy(), plt.gca(), label)

                plt.axis('off')
                output_image_path = os.path.join(vis_output_dir, f"{image_name}_automatic_label_output.jpg")
                plt.savefig(output_image_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
                plt.close()  # 关闭图形，释放内存

                # 保存掩码数据（添加前缀）

                # ====== 新增代码开始：生成并保存 仅包含检测框和标签的图像 ======
                # plt.figure(figsize=(10, 10))
                # plt.imshow(image_cv)  # 只显示原始图像
                #
                # for box, label in zip(boxes_filt, pred_phrases):
                #     show_box(box.numpy(), plt.gca(), label)
                # plt.axis('off')
                # output_image_path_boxes_only = os.path.join(output_dir, f"{image_name}_detection_only.jpg")
                # plt.savefig(output_image_path_boxes_only, bbox_inches="tight", dpi=300, pad_inches=0.0)
                # plt.close()



                save_mask_data(output_dir, image_name,raw_tags, masks, boxes_filt, pred_phrases)

                end = time.time()
                print(f"本张图像处理耗时: {end - start:.2f} 秒")

                print(f"成功处理 {image_file}，检测到 {boxes_filt.shape[0]} 个目标")
            else:
                print(f"警告: {image_file} 未检测到任何目标")

        except Exception as e:
            print(f"处理图像 {image_file} 时出错: {str(e)}")
            continue  # 继续处理下一张图像

    print("\n所有图像处理完成！")
