"""
Evaluate vertebra landmark accuracy on a directory of images with .mat ground truth.
Only evaluates front 28 points (7 vertebrae x 4 corners), ignoring spinous processes.

Usage:
    .venv/Scripts/python evaluate_keypoints.py \
        --image-dir Vertebra-Landmark-Detection/data_renji_vld/data/test \
        --label-dir Vertebra-Landmark-Detection/data_renji_vld/labels/test \
        --resolution-csv Vertebra-Landmark-Detection/data_renji_vld_resolution.csv
"""
import sys
import os
import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import cv2
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'cobb_app', 'src'))
sys.path.insert(0, os.path.join(ROOT, 'Vertebra-Landmark-Detection'))

from inference import load_model, predict_fusion
from vld_inference import VLDModel
from cobb import normalize_corners


def load_resolution_map(path):
    if not path or not os.path.exists(path):
        return {}
    resolution_map = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            name = row[0].strip()
            try:
                resolution_map[name] = np.array([float(row[1]), float(row[2])], dtype=np.float32)
            except ValueError:
                continue
    return resolution_map


def load_gt_mat(mat_path):
    try:
        import scipy.io
        d = scipy.io.loadmat(mat_path)
        p2 = d['p2'].astype(np.float32)
        if p2.shape[0] < 28:
            return None, None
        front = p2[:28, :2].copy()
        back = p2[28:56, :2].copy() if p2.shape[0] >= 56 else None
        return front, back
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return None, None


def unify_gt_orientation(front_pts, back_pts, img_w):
    if back_pts is not None:
        front_x = np.median(front_pts[:, 0])
        back_x = np.median(back_pts[:, 0])
        if front_x > back_x:
            front_pts = front_pts.copy()
            front_pts[:, 0] = img_w - 1 - front_pts[:, 0]
    return front_pts


def sort_vertebrae_by_y(pts_28):
    boxes = [pts_28[i*4:(i+1)*4] for i in range(7)]
    centers = [b.mean(axis=0) for b in boxes]
    sort_idx = np.argsort([c[1] for c in centers])
    sorted_pts = []
    for idx in sort_idx:
        sorted_pts.extend(boxes[idx])
    return np.asarray(sorted_pts, np.float32)


def point_distance_mm(pred, gt, resolution):
    diff = pred - gt
    res = np.asarray(resolution, dtype=np.float32).reshape(-1)
    if res.size >= 2:
        return np.sqrt((diff[0] * res[0])**2 + (diff[1] * res[1])**2)
    return np.linalg.norm(diff) * float(res[0] if res.size else 1.0)


def evaluate(model, config, vld_model, image_dir, label_dir, resolution_map, device='cpu'):
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))

    all_errors_px = []
    all_errors_mm = []
    per_sample = []
    per_vertebra_mm = [[] for _ in range(7)]
    per_corner_mm = [[] for _ in range(28)]

    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        name_no_ext = os.path.splitext(img_name)[0]
        mat_path = os.path.join(label_dir, name_no_ext + '.mat')

        if not os.path.exists(mat_path):
            print(f"[{idx+1}/{len(image_paths)}] SKIP {img_name}: no GT .mat")
            continue

        try:
            pred_pts, orig_image = predict_fusion(model, config, vld_model, img_path, device=device)
        except Exception as e:
            print(f"[{idx+1}/{len(image_paths)}] ERROR {img_name}: inference failed ({e})")
            continue

        h, w = orig_image.shape[:2]

        gt_front, gt_back = load_gt_mat(mat_path)
        if gt_front is None:
            continue

        gt_front = unify_gt_orientation(gt_front, gt_back, w)
        gt_front = sort_vertebrae_by_y(gt_front)
        gt_front = normalize_corners(gt_front)
        pred_pts = normalize_corners(pred_pts)

        res = resolution_map.get(name_no_ext, np.array([1.0, 1.0], dtype=np.float32))

        sample_errors_mm = []
        for i in range(28):
            dist_px = np.linalg.norm(pred_pts[i] - gt_front[i])
            dist_mm = point_distance_mm(pred_pts[i], gt_front[i], res)
            all_errors_px.append(dist_px)
            all_errors_mm.append(dist_mm)
            sample_errors_mm.append(dist_mm)
            v_idx = i // 4
            per_vertebra_mm[v_idx].append(dist_mm)
            per_corner_mm[i].append(dist_mm)

        mean_mm = float(np.mean(sample_errors_mm))
        per_sample.append((name_no_ext, mean_mm))
        print(f"[{idx+1}/{len(image_paths)}] {img_name}: mean={mean_mm:.2f}mm")

    errors_mm = np.asarray(all_errors_mm, dtype=np.float32)
    errors_px = np.asarray(all_errors_px, dtype=np.float32)

    if len(errors_mm) == 0:
        print("No valid samples evaluated.")
        return

    print("\n" + "="*60)
    print("LANDMARK EVALUATION (Front 28 points only)")
    print("="*60)
    print(f"Total images evaluated: {len(per_sample)}")
    print(f"Total points: {len(errors_mm)}")
    print(f"Mean error (px):  {errors_px.mean():.2f}")
    print(f"Mean error (mm):  {errors_mm.mean():.2f}")
    print(f"Std  error (mm):  {errors_mm.std():.2f}")
    print(f"Median error (mm):{np.median(errors_mm):.2f}")
    for thr in [2.0, 2.5, 3.0, 4.0]:
        acc = (errors_mm <= thr).mean()
        print(f"Acc @ {thr}mm:      {acc:.4f}")

    print("\nPer-vertebra mean error (mm):")
    labels = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'T1']
    for i in range(7):
        arr = np.asarray(per_vertebra_mm[i], dtype=np.float32)
        print(f"  {labels[i]}: {arr.mean():.2f}  (n={len(arr)})")

    print("\nPer-corner mean error (mm):")
    corners = ['TL', 'TR', 'BL', 'BR']
    for i in range(28):
        arr = np.asarray(per_corner_mm[i], dtype=np.float32)
        v = labels[i // 4]
        c = corners[i % 4]
        print(f"  {v}_{c}: {arr.mean():.2f}  (n={len(arr)})")

    out_csv = os.path.join(ROOT, 'outputs', 'eval_per_sample.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sample', 'mean_error_mm'])
        for name, err in per_sample:
            writer.writerow([name, f"{err:.4f}"])
    print(f"\nSaved per-sample CSV to: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate vertebra landmark accuracy')
    parser.add_argument('--image-dir', required=True, help='Directory containing PNG images')
    parser.add_argument('--label-dir', required=True, help='Directory containing .mat labels')
    parser.add_argument('--resolution-csv', default='', help='Optional spacing CSV')
    parser.add_argument('--device', default='cpu', help='cuda or cpu')
    args = parser.parse_args()

    cfg_path = os.path.join(ROOT, 'cobb_app', 'config', 'spine_renji_hrnet_w18_pretrained.yaml')
    model_path = os.path.join(ROOT, 'cobb_app', 'models', 'hrnet_renji.pth')
    vld_weights = os.path.join(ROOT, 'cobb_app', 'models', 'spinenet_renji.pth')

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(vld_weights):
        raise FileNotFoundError(f"VLD weights not found: {vld_weights}")

    resolution_map = load_resolution_map(args.resolution_csv)

    print(f"Loading HRNet on {args.device}...")
    model, config = load_model(cfg_path, model_path, device=args.device)
    print("Loading VLD model...")
    vld_model = VLDModel(vld_weights, use_tta=True, device=args.device)

    evaluate(model, config, vld_model, args.image_dir, args.label_dir, resolution_map, device=args.device)


if __name__ == '__main__':
    main()
