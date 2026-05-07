#!/usr/bin/env python3
"""
Generate ensemble visualization by fusing predictions and rendering on images.
"""
import os
import sys
from pathlib import Path

import torch
import numpy as np
import cv2

sys.path.insert(0, os.getcwd())
from ensemble_spine import (
    load_resolution_map, load_hrnet_preds, load_vld_preds, load_dcelr_preds,
    normalize_key, align_samples, align_landmarks_hungarian, fuse_predictions,
)

sys.path.insert(0, 'HRNet-Facial-Landmark-Detection')
from lib.config import config, update_config
from lib.datasets import get_dataset


def load_image(path, dataset_name):
    # HRNet dataset returns npy paths; find original png/jpg
    p = Path(path)
    if p.suffix == '.npy':
        # Try to find matching image in VLD data directory
        img_id = p.stem
        # Remove trailing 'l' or 'm' suffix from npy filenames
        if img_id.endswith('l') or img_id.endswith('m'):
            img_id = img_id[:-1]
        if dataset_name == "RENJI":
            search_dirs = ["Vertebra-Landmark-Detection/data_renji_vld/data/test",
                           "Vertebra-Landmark-Detection/data_renji_vld/data/train"]
        else:
            search_dirs = ["Vertebra-Landmark-Detection/data_ruijin_vld/data/test",
                           "Vertebra-Landmark-Detection/data_ruijin_vld/data/train"]
        for ext in ['.png', '.jpg', '.jpeg']:
            for d in search_dirs:
                candidate = Path(d) / (img_id + ext)
                if candidate.exists():
                    img = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
                    if img is not None:
                        return img
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    raise FileNotFoundError(f"Could not find image for {path}")


def draw_compare(image, gt_pts, pred_pts, radius=4):
    img = image.copy()
    for pt in gt_pts:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img, (x, y), radius, (0, 255, 0), -1)  # green = GT
    for pt in pred_pts:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img, (x, y), radius, (0, 0, 255), -1)  # red = Pred
    return img


def draw_side_by_side(image, gt_pts, pred_pts, radius=4):
    h, w = image.shape[:2]
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    gt_img = image.copy()
    pred_img = image.copy()
    for pt in gt_pts:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(gt_img, (x, y), radius, (0, 255, 0), -1)
    for pt in pred_pts:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(pred_img, (x, y), radius, (0, 0, 255), -1)
    canvas[:, :w] = gt_img
    canvas[:, w:] = pred_img
    return canvas


def generate_ensemble_vis(dataset_name, output_dir, max_samples=8):
    if dataset_name == "RENJI":
        hrnet_path = "HRNet-Facial-Landmark-Detection/outputs/inference_renji_spine_renji_hrnet_w18_pretrained_e60/predictions/predictions_dict.pth"
        vld_path = "Vertebra-Landmark-Detection/outputs/inference_renji_e60_tta/predictions/predictions.pth"
        dcelr_dir = "D-CeLR/outputs/renji_ab_fullimg_e80_improved/eval_best/predictions"
        cfg_file = "HRNet-Facial-Landmark-Detection/experiments/renji/spine_renji_hrnet_w18_pretrained.yaml"
        weights = {"HRNet": 0.4, "VLD": 0.3, "D-CeLR": 0.3}
    else:
        hrnet_path = "HRNet-Facial-Landmark-Detection/outputs/inference_ruijin_spine_ruijin_hrnet_w18_pretrained_e60/predictions/predictions_dict.pth"
        vld_path = "Vertebra-Landmark-Detection/outputs/inference_ruijin_e60/predictions/predictions.pth"
        dcelr_dir = "D-CeLR/outputs/ruijin_ab_fullimg_e80_renji_init/eval_best/predictions"
        cfg_file = "HRNet-Facial-Landmark-Detection/experiments/ruijin/spine_ruijin_hrnet_w18_pretrained.yaml"
        weights = {"HRNet": 0.5, "VLD": 0.5, "D-CeLR": 0.0}

    # Load predictions
    hrnet = load_hrnet_preds(hrnet_path)
    vld = load_vld_preds(vld_path)
    dcelr = load_dcelr_preds(dcelr_dir)

    all_preds = [hrnet, vld, dcelr]
    all_names = ["HRNet", "VLD", "D-CeLR"]
    preds_aligned, common_samples = align_samples(all_preds, all_names)

    # Load GT
    gt_dict = {}
    for p in sorted(Path(dcelr_dir).glob("*_gt.npy")):
        name = normalize_key(p.name.replace("_gt.npy", ""))
        gt_dict[name] = np.load(p)

    # Load dataset for images
    update_config(config, type('Args', (), {'cfg': cfg_file})())
    dataset_type = get_dataset(config)
    dataset = dataset_type(config, is_train=False)

    out_dir = Path(output_dir)
    compare_dir = out_dir / 'compare'
    side_dir = out_dir / 'side_by_side'
    compare_dir.mkdir(parents=True, exist_ok=True)
    side_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx in range(len(dataset)):
        if count >= max_samples:
            break
        _, _, meta = dataset[idx]
        img_id = meta['img_id']
        if img_id not in common_samples:
            continue

        # Get ensemble fused prediction
        ref_idx = all_names.index("VLD")
        ref_pts = preds_aligned[ref_idx][img_id]
        aligned_preds = []
        for i, mp in enumerate(preds_aligned):
            pts = mp[img_id]
            if i == ref_idx:
                aligned_preds.append(pts)
            else:
                aligned_preds.append(align_landmarks_hungarian(pts, ref_pts))

        w_list = [weights[n] for n in all_names]
        fused = fuse_predictions(aligned_preds, weights=w_list, mode="weighted")

        # Visualize
        image = load_image(meta['image_path'], dataset_name)
        gt_pts = meta['pts'].cpu().numpy()
        if gt_pts.shape[0] != fused.shape[0]:
            # Trim or pad to match
            min_pts = min(gt_pts.shape[0], fused.shape[0])
            gt_pts = gt_pts[:min_pts]
            fused = fused[:min_pts]

        compare = draw_compare(image, gt_pts, fused)
        side = draw_side_by_side(image, gt_pts, fused)

        stem = f"{count:03d}_{img_id}"
        cv2.imwrite(str(compare_dir / f'{stem}.png'), compare)
        cv2.imwrite(str(side_dir / f'{stem}.png'), side)
        count += 1
        print(f"Saved {img_id}")

    print(f"Done! {count} samples saved to {output_dir}")


if __name__ == "__main__":
    for ds in ["RENJI", "RUIJIN"]:
        out = f"outputs/best_results/{ds}/ensemble_best"
        print(f"\n=== Generating {ds} ensemble visualization ===")
        generate_ensemble_vis(ds, out, max_samples=8)
