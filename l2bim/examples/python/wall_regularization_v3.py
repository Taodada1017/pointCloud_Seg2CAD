"""
wall_regularization_v3.py - 无 Manhattan 强制约束的墙线合并
保留所有方向的线段，只做共线碎片合并 + 去重
"""
import argparse
import os
import cv2
import numpy as np
from collections import defaultdict

def detect_hough_lines(binary_img, threshold=35, min_length=25, max_gap=8):
    """HoughLinesP 检测"""
    lines = cv2.HoughLinesP(binary_img, 1, np.pi/180, threshold,
                            minLineLength=min_length, maxLineGap=max_gap)
    if lines is None:
        return []
    return [l[0] for l in lines]  # list of [x1,y1,x2,y2]

def line_angle(seg):
    """线段角度 [0, 180)"""
    dx = seg[2] - seg[0]
    dy = seg[3] - seg[1]
    angle = np.degrees(np.arctan2(dy, dx)) % 180
    return angle

def line_length(seg):
    return np.sqrt((seg[2]-seg[0])**2 + (seg[3]-seg[1])**2)

def point_to_line_dist(px, py, x1, y1, x2, y2):
    """点到线段所在直线的距离"""
    dx, dy = x2-x1, y2-y1
    length = np.sqrt(dx*dx + dy*dy)
    if length < 1e-6:
        return np.sqrt((px-x1)**2 + (py-y1)**2)
    return abs(dy*px - dx*py + x2*y1 - y2*x1) / length

def project_on_line(px, py, x1, y1, dx, dy):
    """点在直线方向上的投影参数 t"""
    length = np.sqrt(dx*dx + dy*dy)
    if length < 1e-6:
        return 0
    return ((px-x1)*dx + (py-y1)*dy) / (length*length)

def merge_collinear_segments(segments, angle_thresh=8.0, dist_thresh=15.0, gap_thresh=30.0):
    """
    合并近似共线的碎片线段
    - angle_thresh: 角度差阈值（度）
    - dist_thresh: 垂直距离阈值（像素）
    - gap_thresh: 端点间隙阈值（像素）
    """
    if not segments:
        return []
    
    # 按角度排序
    segs = [(s, line_angle(s), line_length(s)) for s in segments]
    segs.sort(key=lambda x: x[1])
    
    used = [False] * len(segs)
    merged = []
    
    for i in range(len(segs)):
        if used[i]:
            continue
        
        cluster = [segs[i][0]]
        used[i] = True
        ref_angle = segs[i][1]
        
        # 找所有可以合并的线段
        for j in range(i+1, len(segs)):
            if used[j]:
                continue
            
            seg_j = segs[j][0]
            angle_j = segs[j][1]
            
            # 角度差（考虑 0/180 回绕）
            angle_diff = abs(ref_angle - angle_j)
            angle_diff = min(angle_diff, 180 - angle_diff)
            if angle_diff > angle_thresh:
                continue
            
            # 检查垂直距离：seg_j 的两个端点到 cluster 参考线的距离
            ref = cluster[0]
            d1 = point_to_line_dist(seg_j[0], seg_j[1], ref[0], ref[1], ref[2], ref[3])
            d2 = point_to_line_dist(seg_j[2], seg_j[3], ref[0], ref[1], ref[2], ref[3])
            if max(d1, d2) > dist_thresh:
                continue
            
            # 检查间隙：端点之间的最小距离
            min_gap = float('inf')
            for s_existing in cluster:
                for ep1 in [(seg_j[0], seg_j[1]), (seg_j[2], seg_j[3])]:
                    for ep2 in [(s_existing[0], s_existing[1]), (s_existing[2], s_existing[3])]:
                        d = np.sqrt((ep1[0]-ep2[0])**2 + (ep1[1]-ep2[1])**2)
                        min_gap = min(min_gap, d)
            
            # 如果端点很近或有重叠，合并
            if min_gap < gap_thresh:
                cluster.append(seg_j)
                used[j] = True
        
        # 用 fitLine 拟合 cluster 中所有点，取 t 的范围
        pts = []
        for s in cluster:
            pts.append([s[0], s[1]])
            pts.append([s[2], s[3]])
        pts = np.array(pts, dtype=np.float32)
        
        if len(pts) < 2:
            merged.append(cluster[0])
            continue
        
        # 拟合直线
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)
        
        # 投影所有点到直线上，取极值
        t_vals = []
        for p in pts:
            t = (p[0]-x0)*vx + (p[1]-y0)*vy
            t_vals.append(t)
        
        t_min, t_max = min(t_vals), max(t_vals)
        
        x1 = int(round(x0 + t_min * vx))
        y1 = int(round(y0 + t_min * vy))
        x2 = int(round(x0 + t_max * vx))
        y2 = int(round(y0 + t_max * vy))
        
        merged.append([x1, y1, x2, y2])
    
    return merged

def remove_short_segments(segments, min_length=20):
    return [s for s in segments if line_length(s) >= min_length]

def remove_duplicate_segments(segments, angle_thresh=5.0, dist_thresh=8.0, overlap_thresh=0.7):
    """去除重复/高度重叠的线段，保留较长的"""
    if not segments:
        return []
    
    segs = sorted(segments, key=lambda s: -line_length(s))  # 长的优先
    keep = [True] * len(segs)
    
    for i in range(len(segs)):
        if not keep[i]:
            continue
        for j in range(i+1, len(segs)):
            if not keep[j]:
                continue
            
            angle_diff = abs(line_angle(segs[i]) - line_angle(segs[j]))
            angle_diff = min(angle_diff, 180 - angle_diff)
            if angle_diff > angle_thresh:
                continue
            
            # 短线段端点到长线段的距离
            d1 = point_to_line_dist(segs[j][0], segs[j][1], 
                                     segs[i][0], segs[i][1], segs[i][2], segs[i][3])
            d2 = point_to_line_dist(segs[j][2], segs[j][3],
                                     segs[i][0], segs[i][1], segs[i][2], segs[i][3])
            if max(d1, d2) < dist_thresh:
                keep[j] = False
    
    return [s for s, k in zip(segs, keep) if k]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Binary line image (e.g. projection_lines_refined.png)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hough-threshold", type=int, default=30)
    parser.add_argument("--hough-min-length", type=int, default=20)
    parser.add_argument("--hough-max-gap", type=int, default=10)
    parser.add_argument("--merge-angle", type=float, default=8.0)
    parser.add_argument("--merge-dist", type=float, default=15.0)
    parser.add_argument("--merge-gap", type=float, default=35.0)
    parser.add_argument("--min-length", type=int, default=20)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: Cannot read {args.input}")
        return
    
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    h, w = binary.shape
    print(f"Image size: {w}x{h}")
    print(f"Non-zero pixels: {np.count_nonzero(binary)}")
    
    # Step 0: 保存输入
    cv2.imwrite(os.path.join(args.output_dir, "v3_step0_input.png"), binary)
    print("  ✓ v3_step0_input.png")
    
    # Step 1: Hough 检测
    raw_lines = detect_hough_lines(binary, args.hough_threshold, 
                                    args.hough_min_length, args.hough_max_gap)
    print(f"\n[Step 1] Hough detection: {len(raw_lines)} raw segments")
    
    vis_raw = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for seg in raw_lines:
        cv2.line(vis_raw, (seg[0], seg[1]), (seg[2], seg[3]), (0, 0, 255), 1)
    cv2.imwrite(os.path.join(args.output_dir, "v3_step1_hough_raw.png"), vis_raw)
    print("  ✓ v3_step1_hough_raw.png")
    
    # Step 2: 合并共线碎片
    merged = merge_collinear_segments(raw_lines, args.merge_angle, 
                                       args.merge_dist, args.merge_gap)
    print(f"\n[Step 2] Collinear merge: {len(raw_lines)} → {len(merged)} segments")
    
    vis_merged = np.zeros((h, w, 3), dtype=np.uint8)
    for seg in merged:
        cv2.line(vis_merged, (seg[0], seg[1]), (seg[2], seg[3]), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(args.output_dir, "v3_step2_merged.png"), vis_merged)
    print("  ✓ v3_step2_merged.png")
    
    # Step 3: 去除短线段
    filtered = remove_short_segments(merged, args.min_length)
    print(f"\n[Step 3] Remove short (<{args.min_length}px): {len(merged)} → {len(filtered)}")
    
    # Step 4: 去重
    deduped = remove_duplicate_segments(filtered)
    print(f"\n[Step 4] Deduplicate: {len(filtered)} → {len(deduped)}")
    
    # Step 5: 最终结果（黑底白线）
    final = np.zeros((h, w), dtype=np.uint8)
    for seg in deduped:
        cv2.line(final, (seg[0], seg[1]), (seg[2], seg[3]), 255, 2)
    cv2.imwrite(os.path.join(args.output_dir, "v3_step5_final.png"), final)
    print(f"\n  ✓ v3_step5_final.png — {len(deduped)} wall segments")
    
    # Step 6: 彩色对比（红=原始Hough，绿=合并后）
    vis_compare = np.zeros((h, w, 3), dtype=np.uint8)
    for seg in raw_lines:
        cv2.line(vis_compare, (seg[0], seg[1]), (seg[2], seg[3]), (0, 0, 255), 1)
    for seg in deduped:
        cv2.line(vis_compare, (seg[0], seg[1]), (seg[2], seg[3]), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(args.output_dir, "v3_step6_comparison.png"), vis_compare)
    print("  ✓ v3_step6_comparison.png (Red=raw, Green=merged)")
    
    # Step 7: 叠加在投影图上
    # 尝试读取 projection.png 做叠加
    proj_path = os.path.join(args.output_dir, "projection.png")
    if os.path.exists(proj_path):
        proj = cv2.imread(proj_path)
        if proj is not None:
            overlay = proj.copy()
            for seg in deduped:
                cv2.line(overlay, (seg[0], seg[1]), (seg[2], seg[3]), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(args.output_dir, "v3_step7_overlay.png"), overlay)
            print("  ✓ v3_step7_overlay.png (lines on projection)")
    
    print(f"\n{'='*60}")
    print(f"Done! {len(deduped)} final wall segments (no Manhattan constraint)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
