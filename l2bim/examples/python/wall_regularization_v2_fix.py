import cv2
import numpy as np
import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class LineSegment:
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def angle(self):
        angle = np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1))
        if angle < 0:
            angle += 180
        return angle
    
    @property
    def length(self):
        return np.hypot(self.x2 - self.x1, self.y2 - self.y1)
    
    @property
    def direction_vector(self):
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        length = self.length
        if length < 1:
            return (0, 0)
        # 曼哈顿世界假设：15度容差内吸附到水平/垂直方向
        angle = self.angle  # 0~180
        snap_tolerance = 15
        if angle < snap_tolerance or angle > (180 - snap_tolerance):
            # 接近水平 (0° / 180°)，强制水平方向
            return (1.0 if dx >= 0 else -1.0, 0.0)
        elif abs(angle - 90) < snap_tolerance:
            # 接近垂直 (90°)，强制垂直方向
            return (0.0, 1.0 if dy >= 0 else -1.0)
        else:
            return (dx / length, dy / length)
    
    def get_points(self):
        return (self.x1, self.y1), (self.x2, self.y2)

BLOCK_SIZE = 30
FIT_ERROR_THRESHOLD = 5.0
EXTEND_STEP = 2
EXTEND_MAX_DISTANCE = 300
IRREGULAR_CHECK_RADIUS = 8
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 20
HOUGH_MIN_LINE_LENGTH = 30
HOUGH_MAX_LINE_GAP = 15

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = current_dir

def save_image(img, filename, description):
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, img)
    print("  \u2713 已保存: " + filename + " - " + description)

def load_image(input_path=None):
    if input_path is None:
        input_path = os.path.join(current_dir, "projection_plus.png")
    img = cv2.imread(input_path, 0)
    if img is None:
        raise FileNotFoundError("Cannot find input image: " + str(input_path))
    return img

def parse_args():
    parser = argparse.ArgumentParser(description="Indoor Wall Line Regularization v2.0")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(current_dir, "projection_plus.png"),
        help="Input projection image path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=current_dir,
        help="Output directory for result images"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=BLOCK_SIZE,
        help="Block size for regular/irregular region classification"
    )
    parser.add_argument(
        "--min-white-ratio",
        type=float,
        default=0.3,
        help="Minimum white pixel ratio required for path validation"
    )
    parser.add_argument(
        "--merge-dist-thresh",
        type=float,
        default=25.0,
        help="Base max offset distance for merging parallel segments"
    )
    parser.add_argument(
        "--merge-gap-thresh",
        type=float,
        default=50.0,
        help="Base max gap distance for merging collinear fragments"
    )
    parser.add_argument(
        "--merge-hard-dist-cap",
        type=float,
        default=35.0,
        help="Hard cap: never merge parallel segments farther than this distance"
    )
    parser.add_argument(
        "--merge-busy-trigger",
        type=int,
        default=120,
        help="If base segment count reaches this value, enable stronger merge"
    )
    parser.add_argument(
        "--merge-busy-scale",
        type=float,
        default=1.35,
        help="Threshold scale applied when segment count is high"
    )
    return parser.parse_args()

def align_manhattan_image(binary_img):
    edges = cv2.Canny(binary_img, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=30,
        maxLineGap=10,
    )

    if lines is None:
        return binary_img, 0.0

    offset_angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        offset_angle = (angle + 45.0) % 90.0 - 45.0
        offset_angles.append(float(offset_angle))

    if len(offset_angles) == 0:
        return binary_img, 0.0

    dominant_angle = float(np.median(np.asarray(offset_angles, dtype=np.float32)))
    h, w = binary_img.shape
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, dominant_angle, 1.0)
    aligned_img = cv2.warpAffine(
        binary_img,
        rot_mat,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return aligned_img, dominant_angle

def divide_image_into_blocks(h, w):
    blocks = []
    for y in range(0, h, BLOCK_SIZE):
        for x in range(0, w, BLOCK_SIZE):
            end_y = min(y + BLOCK_SIZE, h)
            end_x = min(x + BLOCK_SIZE, w)
            blocks.append((x, y, end_x, end_y))
    return blocks

def check_block_regularity(points, error_threshold=2.0):
    """使用 cv2.fitLine 的正交距离误差判断区域是否规则。"""
    if len(points) < 2:
        return False

    # points: (row, col) -> (x, y)，并转换为 fitLine 需要的 (-1, 1, 2)
    pts_xy = points[:, ::-1].astype(np.float32).reshape(-1, 1, 2)

    vx, vy, x0, y0 = cv2.fitLine(pts_xy, cv2.DIST_L2, 0, 0.01, 0.01)
    vx = float(vx)
    vy = float(vy)
    x0 = float(x0)
    y0 = float(y0)

    pts = pts_xy.reshape(-1, 2)
    px = pts[:, 0]
    py = pts[:, 1]

    # 点到直线的法向距离: |(p - p0) x v| / |v|
    denom = np.hypot(vx, vy) + 1e-8
    distances = np.abs((px - x0) * vy - (py - y0) * vx) / denom
    mean_dist = float(np.mean(distances))

    return mean_dist < error_threshold

def is_path_valid(img, p1, p2, min_white_ratio=0.30):
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.line(mask, p1, p2, 255, 3)

    line_region = mask > 0
    total_pixels = int(np.count_nonzero(line_region))
    if total_pixels == 0:
        return False

    white_pixels = int(np.count_nonzero(img[line_region] > 128))
    white_ratio = white_pixels / float(total_pixels)
    return white_ratio >= min_white_ratio

def mark_regions(binary_img, output_vis=True):
    h, w = binary_img.shape
    region_mask = np.zeros((h, w), dtype=np.uint8)
    regular_mask = np.zeros((h, w), dtype=np.uint8)

    # 取消规则区域筛选：直接将全部前景作为霍夫输入，避免早期误筛。
    foreground = binary_img > 128
    region_mask[foreground] = 1
    regular_mask[foreground] = binary_img[foreground]
    
    if output_vis:
        region_vis = np.zeros((h, w, 3), dtype=np.uint8)
        region_vis[region_mask == 0] = [50, 50, 50]
        region_vis[region_mask == 1] = [0, 255, 0]
        region_vis[region_mask == 2] = [0, 0, 255]
        save_image(region_vis, "v2_step1_region_mask.png", "Region Mask (Green=Regular, Red=Irregular)")
        
        regular_vis = cv2.cvtColor(regular_mask, cv2.COLOR_GRAY2BGR)
        save_image(regular_vis, "v2_step2_regular_regions.png", "Regular Regions Extracted")
    
    return region_mask, regular_mask

def hough_lines_on_regions(regular_mask, original_img):
    h, w = regular_mask.shape
    
    # 骨架化前先做闭开平滑：先填洞，再抹平锯齿
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    smoothed_mask = cv2.morphologyEx(regular_mask, cv2.MORPH_CLOSE, smooth_kernel)

    # 直接在平滑后的掩膜上做边缘提取，避免骨架化带来的毛刺断裂
    edges = cv2.Canny(smoothed_mask, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 
                            rho=HOUGH_RHO, 
                            theta=HOUGH_THETA, 
                            threshold=HOUGH_THRESHOLD,
                            minLineLength=HOUGH_MIN_LINE_LENGTH,
                            maxLineGap=HOUGH_MAX_LINE_GAP)
    
    segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            segments.append(LineSegment(int(x1), int(y1), int(x2), int(y2)))
    
    hough_vis = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    for seg in segments:
        cv2.line(hough_vis, (seg.x1, seg.y1), (seg.x2, seg.y2), (0, 255, 0), 2)
    save_image(hough_vis, "v2_step3_hough_lines.png", "Hough Lines in Regular Regions")
    
    return segments

def merge_collinear_segments(segments, binary_img, dist_thresh=25, gap_thresh=50, hard_dist_cap=35, min_white_ratio=0.3):
    horizontal = []
    vertical = []

    for seg in segments:
        dx = abs(seg.x2 - seg.x1)
        dy = abs(seg.y2 - seg.y1)
        if dx >= dy:
            x_min, x_max = sorted([seg.x1, seg.x2])
            y_avg = int(round((seg.y1 + seg.y2) / 2.0))
            horizontal.append([x_min, x_max, y_avg])
        else:
            y_min, y_max = sorted([seg.y1, seg.y2])
            x_avg = int(round((seg.x1 + seg.x2) / 2.0))
            vertical.append([y_min, y_max, x_avg])

    def interval_gap(a1, a2, b1, b2):
        if a2 < b1:
            return b1 - a2
        if b2 < a1:
            return a1 - b2
        return 0

    def get_interval_bridge(a1, a2, b1, b2):
        if a2 < b1:
            return a2, b1
        if b2 < a1:
            return b2, a1
        return None

    def get_segment_length(line):
        return abs(line[1] - line[0])

    def merge_fragments(lines, is_horizontal=True):
        """第一阶段：只合并同一条直线上的碎片段，不跨较远平行线。"""
        if len(lines) <= 1:
            return lines

        # 偏移容差小：仅允许同线碎片合并；间隙容差大一些：允许碎片补全成长线。
        offset_tol = max(1, min(4, int(round(dist_thresh * 0.2))))
        fragment_gap_tol = max(6, min(int(round(gap_thresh * 0.7)), 40))

        merged = lines[:]
        changed = True
        while changed:
            changed = False
            used = [False] * len(merged)
            new_merged = []

            for i in range(len(merged)):
                if used[i]:
                    continue
                a1, a2, c = merged[i]
                used[i] = True

                local_changed = True
                while local_changed:
                    local_changed = False
                    for j in range(len(merged)):
                        if used[j]:
                            continue

                        b1, b2, oc = merged[j]
                        if abs(c - oc) > offset_tol:
                            continue

                        gap = interval_gap(a1, a2, b1, b2)
                        if gap > fragment_gap_tol:
                            continue

                        # 若两段在主轴上明显重叠且法向有偏移，通常是墙两侧平行边，禁止合并。
                        overlap = max(0, min(a2, b2) - max(a1, b1))
                        if overlap > 0 and abs(c - oc) >= 1:
                            continue

                        candidate_c = int(round((c + oc) / 2.0))
                        bridge = get_interval_bridge(a1, a2, b1, b2)
                        if bridge is not None:
                            if is_horizontal:
                                p1 = (int(bridge[0]), candidate_c)
                                p2 = (int(bridge[1]), candidate_c)
                            else:
                                p1 = (candidate_c, int(bridge[0]))
                                p2 = (candidate_c, int(bridge[1]))
                            if not is_path_valid(binary_img, p1, p2, min_white_ratio=min_white_ratio):
                                continue

                        a1 = min(a1, b1)
                        a2 = max(a2, b2)
                        c = candidate_c
                        used[j] = True
                        local_changed = True
                        changed = True

                new_merged.append([a1, a2, c])
            merged = new_merged

        return merged

    def normalize_parallel_lines(lines):
        """第二阶段：局部平行线规范化。每个局部簇最多保留2条（墙两侧），抑制第3条及以上干扰。"""
        if len(lines) <= 2:
            return lines

        # 沿法向坐标聚类，避免全局误删。
        coord_cluster_tol = max(6, min(int(round(hard_dist_cap * 0.45)), 16))
        # 沿主方向要求有空间邻近，避免跨房间互相竞争。
        main_axis_group_gap = max(10, min(int(round(gap_thresh * 1.2)), 70))

        by_coord = sorted(lines, key=lambda ln: ln[2])
        coord_clusters = []
        cur_cluster = [by_coord[0]]
        for ln in by_coord[1:]:
            if abs(ln[2] - cur_cluster[-1][2]) <= coord_cluster_tol:
                cur_cluster.append(ln)
            else:
                coord_clusters.append(cur_cluster)
                cur_cluster = [ln]
        coord_clusters.append(cur_cluster)

        kept = []
        for cluster in coord_clusters:
            # 在同一法向簇内，再按主轴位置分局部组。
            cluster_sorted = sorted(cluster, key=lambda ln: ln[0])
            local_groups = []
            cur_group = [cluster_sorted[0]]
            group_end = cluster_sorted[0][1]
            for ln in cluster_sorted[1:]:
                if interval_gap(cur_group[0][0], group_end, ln[0], ln[1]) <= main_axis_group_gap:
                    cur_group.append(ln)
                    group_end = max(group_end, ln[1])
                else:
                    local_groups.append(cur_group)
                    cur_group = [ln]
                    group_end = ln[1]
            local_groups.append(cur_group)

            for group in local_groups:
                if len(group) <= 2:
                    kept.extend(group)
                    continue
                group_sorted = sorted(group, key=get_segment_length, reverse=True)
                kept.extend(group_sorted[:2])

        return kept

    merged_horizontal = merge_fragments(horizontal, is_horizontal=True)
    merged_vertical = merge_fragments(vertical, is_horizontal=False)

    filtered_horizontal = normalize_parallel_lines(merged_horizontal)
    filtered_vertical = normalize_parallel_lines(merged_vertical)

    result = []
    for x1, x2, y in filtered_horizontal:
        result.append(LineSegment(int(x1), int(y), int(x2), int(y)))
    for y1, y2, x in filtered_vertical:
        result.append(LineSegment(int(x), int(y1), int(x), int(y2)))

    dedup = []
    seen = set()
    for seg in result:
        key = (seg.x1, seg.y1, seg.x2, seg.y2)
        if key not in seen:
            seen.add(key)
            dedup.append(seg)
    return dedup

def intersect_orthogonal_segments(segments, binary_img, snap_radius=50, min_white_ratio=0.3):
    processed = [LineSegment(s.x1, s.y1, s.x2, s.y2) for s in segments]

    horizontal_indices = []
    vertical_indices = []
    for idx, seg in enumerate(processed):
        if abs(seg.x2 - seg.x1) >= abs(seg.y2 - seg.y1):
            horizontal_indices.append(idx)
        else:
            vertical_indices.append(idx)

    def get_h_endpoints(seg):
        if seg.x1 <= seg.x2:
            return ("left", (seg.x1, seg.y1)), ("right", (seg.x2, seg.y2))
        return ("left", (seg.x2, seg.y2)), ("right", (seg.x1, seg.y1))

    def get_v_endpoints(seg):
        if seg.y1 <= seg.y2:
            return ("top", (seg.x1, seg.y1)), ("bottom", (seg.x2, seg.y2))
        return ("top", (seg.x2, seg.y2)), ("bottom", (seg.x1, seg.y1))

    def set_h_endpoint(seg, side, x, y):
        if (side == "left" and seg.x1 <= seg.x2) or (side == "right" and seg.x1 > seg.x2):
            seg.x1, seg.y1 = x, y
        else:
            seg.x2, seg.y2 = x, y
        avg_y = int(round((seg.y1 + seg.y2) / 2.0))
        seg.y1 = avg_y
        seg.y2 = avg_y

    def set_v_endpoint(seg, side, x, y):
        if (side == "top" and seg.y1 <= seg.y2) or (side == "bottom" and seg.y1 > seg.y2):
            seg.x1, seg.y1 = x, y
        else:
            seg.x2, seg.y2 = x, y
        avg_x = int(round((seg.x1 + seg.x2) / 2.0))
        seg.x1 = avg_x
        seg.x2 = avg_x

    for hi in horizontal_indices:
        hseg = processed[hi]
        for vi in vertical_indices:
            vseg = processed[vi]

            (h_left_label, h_left_pt), (h_right_label, h_right_pt) = get_h_endpoints(hseg)
            (v_top_label, v_top_pt), (v_bottom_label, v_bottom_pt) = get_v_endpoints(vseg)

            pairs = [
                (h_left_label, h_left_pt, v_top_label, v_top_pt),
                (h_left_label, h_left_pt, v_bottom_label, v_bottom_pt),
                (h_right_label, h_right_pt, v_top_label, v_top_pt),
                (h_right_label, h_right_pt, v_bottom_label, v_bottom_pt),
            ]

            min_dist = float("inf")
            best_pair = None
            for hs, hp, vs, vp in pairs:
                dist = np.hypot(hp[0] - vp[0], hp[1] - vp[1])
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (hs, hp, vs, vp)

            if best_pair is None or min_dist >= snap_radius:
                continue

            intersection_x = int(round((vseg.x1 + vseg.x2) / 2.0))
            intersection_y = int(round((hseg.y1 + hseg.y2) / 2.0))

            _, h_endpoint, _, v_endpoint = best_pair
            intersection_point = (intersection_x, intersection_y)
            if not is_path_valid(binary_img, h_endpoint, intersection_point, min_white_ratio=min_white_ratio):
                continue
            if not is_path_valid(binary_img, v_endpoint, intersection_point, min_white_ratio=min_white_ratio):
                continue

            set_h_endpoint(hseg, best_pair[0], intersection_x, intersection_y)
            set_v_endpoint(vseg, best_pair[2], intersection_x, intersection_y)

    return processed

def main():
    global output_dir, BLOCK_SIZE
    args = parse_args()
    output_dir = args.output_dir
    BLOCK_SIZE = max(4, args.block_size)
    min_white_ratio = min(max(args.min_white_ratio, 0.0), 1.0)
    merge_dist_thresh = max(1.0, args.merge_dist_thresh)
    merge_gap_thresh = max(0.0, args.merge_gap_thresh)
    merge_hard_dist_cap = max(1.0, args.merge_hard_dist_cap)
    merge_busy_trigger = max(1, args.merge_busy_trigger)
    merge_busy_scale = max(1.0, args.merge_busy_scale)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Indoor Wall Line Regularization v2.0")
    print("=" * 70)
    
    raw_img = load_image(args.input)
    h, w = raw_img.shape
    print("Image size: " + str(h) + "x" + str(w))
    print("Block size: " + str(BLOCK_SIZE))
    print("Min white ratio: " + str(min_white_ratio))
    print("Merge base dist/gap: " + str(merge_dist_thresh) + "/" + str(merge_gap_thresh))
    print("Merge hard dist cap: " + str(merge_hard_dist_cap))
    print("Merge busy trigger/scale: " + str(merge_busy_trigger) + "/" + str(merge_busy_scale))

    print("\n[Step 0.5] Manhattan auto-alignment...")
    binary_img, tilt_angle = align_manhattan_image(raw_img)
    print(f"Auto-aligned image by {tilt_angle:.2f} degrees")

    save_image(cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR), "v2_step0_raw_input.png", "Raw Input Before Alignment")
    save_image(cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR), "v2_step0_input.png", "Aligned Input Image")
    save_image(cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR), "v2_step0_aligned_input.png", "Input After Wall Alignment")
    
    print("\n[Step 1] Marking regular/irregular regions...")
    region_mask, regular_mask = mark_regions(binary_img, output_vis=True)
    
    print("\n[Step 2] Hough line detection in regular regions...")
    segments = hough_lines_on_regions(regular_mask, binary_img)
    print("  Detected " + str(len(segments)) + " base line segments")

    if len(segments) >= merge_busy_trigger:
        effective_dist_thresh = min(merge_dist_thresh * merge_busy_scale, merge_hard_dist_cap)
        effective_gap_thresh = merge_gap_thresh * merge_busy_scale
        print("  Busy scene detected, merge thresholds scaled up")
    else:
        effective_dist_thresh = min(merge_dist_thresh, merge_hard_dist_cap)
        effective_gap_thresh = merge_gap_thresh

    print("  Effective merge dist/gap: " + str(effective_dist_thresh) + "/" + str(effective_gap_thresh))
    
    print("\n[Step 3] Geometric topology regularization...")
    # 停止生长：本轮实验直接使用霍夫线，不做合并/拓扑延伸。
    final_segments = [LineSegment(s.x1, s.y1, s.x2, s.y2) for s in segments]
    print("  Growth stopped, keeping Hough segments: " + str(len(final_segments)))
    
    print("\n[Step 4] Generating final results...")
    
    out_original = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    for seg in segments:
        cv2.line(out_original, (seg.x1, seg.y1), (seg.x2, seg.y2), (0, 255, 0), 2)
    save_image(out_original, "v2_step5_original_lines.png", "Original Hough Lines")
    
    out_extended = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    for seg in final_segments:
        cv2.line(out_extended, (seg.x1, seg.y1), (seg.x2, seg.y2), (0, 255, 0), 2)
    save_image(out_extended, "v2_step6_extended_lines.png", "Extended Regularized Lines")
    
    out_combined = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    for seg in segments:
        cv2.line(out_combined, (seg.x1, seg.y1), (seg.x2, seg.y2), (0, 0, 255), 1)
    for seg in final_segments:
        cv2.line(out_combined, (seg.x1, seg.y1), (seg.x2, seg.y2), (0, 255, 0), 2)
    save_image(out_combined, "v2_step7_combined.png", "Comparison (Red=Original, Green=Extended)")
    
    out_final = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    for seg in final_segments:
        cv2.line(out_final, (seg.x1, seg.y1), (seg.x2, seg.y2), (0, 255, 0), 3)
    save_image(out_final, "v2_step8_final_result.png", "Final Regularized Result")
    
    print("\n" + "=" * 70)
    print("Processing complete! Generated intermediate files:")
    print("  1. v2_step0_raw_input.png    - Raw Input Before Alignment")
    print("  2. v2_step0_input.png        - Aligned Input Image")
    print("  3. v2_step0_aligned_input.png - Input After Wall Alignment")
    print("  4. v2_step1_region_mask.png  - Region Mask (Green=Regular, Red=Irregular)")
    print("  5. v2_step2_regular_regions.png - Regular Regions Extracted")
    print("  6. v2_step3_hough_lines.png  - Hough Lines in Regular Regions")
    print("  7. v2_step5_original_lines.png - Original Hough Lines")
    print("  8. v2_step6_extended_lines.png - Extended Regularized Lines")
    print("  9. v2_step7_combined.png     - Original vs Extended Comparison")
    print(" 10. v2_step8_final_result.png - Final Regularized Result")
    print("=" * 70)
    
    cv2.imshow("Region Mask", cv2.imread(os.path.join(output_dir, "v2_step1_region_mask.png")))
    cv2.imshow("Hough Lines", cv2.imread(os.path.join(output_dir, "v2_step3_hough_lines.png")))
    cv2.imshow("Final Result", cv2.imread(os.path.join(output_dir, "v2_step8_final_result.png")))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
