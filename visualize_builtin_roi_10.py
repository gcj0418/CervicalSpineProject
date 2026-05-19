#!/usr/bin/env python3
"""
Visualize 5 best + 5 worst nnU-Net ROI predictions on test set.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
HRNET_VLD_ROOT = PROJECT_ROOT / "hrnet-vld_portable"
DATASET_ROOT = PROJECT_ROOT / "D-CeLR"
RESOLUTION_CSV = PROJECT_ROOT / "D-CeLR" / "data" / "renji_npy_direct" / "test_resolution.csv"
RENJI_RAW_ROOT = PROJECT_ROOT / "data" / "RENJI"
OUTPUT_VIS_DIR = PROJECT_ROOT / "outputs" / "builtin_roi_vis_10"

sys.path.insert(0, str(HRNET_VLD_ROOT))

from inference_with_roi import HRNetROIInference
from inference.roi_predictor import BuiltinROIPredictor
from visualization.draw import draw_prediction

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise ImportError("scipy is required") from exc

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("SimpleITK is required") from exc


# 5 best + 5 worst from previous evaluation
SELECTED_SAMPLES = {
    # Best 5
    "HUANG_LI_PING_DX_Cervical-spine_20240924_1": {"category": "best", "err": 1.303},
    "HUANG_HUI_QIN_CR_Cervical_spine_lat_20230410": {"category": "best", "err": 1.529},
    "HE_MAO_FANG_DX__20240829": {"category": "best", "err": 1.701},
    "HE_XIAO_FANG_DX__20250103_2": {"category": "best", "err": 1.986},
    "HE_XIAO_FANG_DX__20250103": {"category": "best", "err": 2.154},
    # Worst 5
    "JIANG_SHU_LIN_CR_Cervical_spine_lat_20211129_1": {"category": "worst", "err": 15.789},
    "ZHOU_HONG_FU_CR_Cervical_spine_lat_20230208_2": {"category": "worst", "err": 14.399},
    "JIN_BAO_HONG_DX__20240122_1": {"category": "worst", "err": 13.105},
    "ZHOU_HONG_FU_CR_Cervical_spine_lat_20230208": {"category": "worst", "err": 11.762},
    "HUANG_XU_DONG_DX__20240701": {"category": "worst", "err": 11.063},
}


def load_resolution_map(path: Path) -> dict[str, tuple[float, float]]:
    res_map = {}
    if not path.exists():
        return res_map
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            img_id = row[0].strip()
            sx = float(row[1])
            sy = float(row[2]) if len(row) >= 3 else sx
            res_map[img_id] = (sx, sy)
    return res_map


def get_resolution(res_map: dict, img_id: str) -> tuple[float, float]:
    if img_id in res_map:
        return res_map[img_id]
    base = img_id.rsplit("_", 1)[0] if "_" in img_id else img_id
    if base in res_map:
        return res_map[base]
    return (1.0, 1.0)


def find_original_niigz(img_id: str) -> Path | None:
    search_name = f"{img_id}.nii.gz"
    for root, dirs, files in os.walk(RENJI_RAW_ROOT):
        for f in files:
            if f == search_name:
                return Path(root) / f
    return None


def roi_from_mask(mask_path: Path, fallback_shape: tuple[int, int]) -> tuple[np.ndarray, float] | None:
    mask = sitk.ReadImage(str(mask_path))
    mask_array = sitk.GetArrayFromImage(mask)
    if mask_array.ndim == 3:
        mask_2d = mask_array[mask_array.shape[0] // 2]
    elif mask_array.ndim == 2:
        mask_2d = mask_array
    else:
        return None

    coords = np.argwhere(mask_2d > 0)
    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    center = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=np.float32)

    width = max(float(x_max - x_min + 1), 1.0)
    height = max(float(y_max - y_min + 1), 1.0)
    longest_side = max(width, height)
    scale = float(longest_side / 200.0 * 1.35)

    h, w = fallback_shape
    center[0] = float(np.clip(center[0], 0, w - 1))
    center[1] = float(np.clip(center[1], 0, h - 1))
    return center, scale


def hungarian_error_mm(pred_pts: np.ndarray, gt_pts: np.ndarray, resolution: tuple[float, float]) -> np.ndarray:
    pred_pts = np.asarray(pred_pts, dtype=np.float32)
    gt_pts = np.asarray(gt_pts, dtype=np.float32)
    sx, sy = resolution
    spacing = np.array([sx, sy], dtype=np.float32)
    pred_mm = pred_pts * spacing
    gt_mm = gt_pts * spacing
    cost = np.linalg.norm(pred_mm[:, None, :] - gt_mm[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return cost[row_ind, col_ind].astype(np.float32)


def draw_roi_rect(image: np.ndarray, center: np.ndarray, scale: float, color: tuple = (0, 255, 255), thickness: int = 2) -> np.ndarray:
    canvas = image.copy()
    size = scale * 200
    half = size / 2
    x1 = int(center[0] - half)
    y1 = int(center[1] - half)
    x2 = int(center[0] + half)
    y2 = int(center[1] + half)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    return canvas


def visualize_sample(
    image_bgr: np.ndarray,
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    center: np.ndarray,
    scale: float,
    mean_err: float,
    category: str,
    out_path: Path,
) -> None:
    canvas = draw_prediction(image_bgr, pred_pts, show_boxes=True, show_keypoints=True, show_labels=True)

    # Draw GT keypoints in green
    for pt in gt_pts:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(canvas, (x, y), 3, (0, 255, 0), -1, cv2.LINE_AA)

    # Draw ROI rectangle
    canvas = draw_roi_rect(canvas, center, scale, color=(0, 255, 255), thickness=2)

    # Add error text
    h, w = canvas.shape[:2]
    text = f"[{category}] mean_err={mean_err:.2f}mm"
    cv2.putText(canvas, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)


def main():
    print("=" * 60)
    print("Visualizing 5 best + 5 worst nnU-Net ROI predictions")
    print("=" * 60)

    OUTPUT_VIS_DIR.mkdir(parents=True, exist_ok=True)

    # Find original .nii.gz for selected samples
    nii_map: dict[str, Path] = {}
    for img_id in SELECTED_SAMPLES:
        nii_path = find_original_niigz(img_id)
        if nii_path is not None:
            nii_map[img_id] = nii_path
        else:
            print(f"[WARN] Original .nii.gz not found for {img_id}")

    print(f"Found {len(nii_map)}/{len(SELECTED_SAMPLES)} original .nii.gz files")

    if len(nii_map) == 0:
        print("No files found, aborting.")
        return

    # Init models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hrnet_cfg = HRNET_VLD_ROOT / "experiments" / "renji" / "spine_renji_hrnet_w18_pretrained.yaml"
    hrnet_weights = HRNET_VLD_ROOT / "weights" / "model_best.pth"
    print(f"Loading HRNet...")
    hrnet = HRNetROIInference(cfg_path=str(hrnet_cfg), model_path=str(hrnet_weights), device=device)
    print(f"Loading nnU-Net ROI predictor...")
    roi_predictor = BuiltinROIPredictor(device=device)

    res_map = load_resolution_map(RESOLUTION_CSV)

    # Prepare temp dir with symlinks
    temp_nii_dir = Path(tempfile.mkdtemp(prefix="renji_nii_vis10_"))
    nii_paths: list[Path] = []
    for img_id in nii_map:
        src = nii_map[img_id]
        dst = temp_nii_dir / f"{img_id}.nii.gz"
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            import shutil
            shutil.copy2(src, dst)
        nii_paths.append(dst)

    # Run nnU-Net prediction for all 10 at once
    mask_output_dir = temp_nii_dir / "roi_masks"
    print(f"Running nnU-Net prediction on {len(nii_paths)} samples...")
    mask_paths = roi_predictor.predict_masks(nii_paths, mask_output_dir)

    roi_dict: dict[str, dict] = {}
    for img_id in nii_map:
        mask_path = mask_paths.get(img_id)
        if mask_path is None:
            print(f"[WARN] No mask for {img_id}")
            continue
        npy_path = DATASET_ROOT / f"data/renji_npy_direct/test/{img_id}m.npy"
        if not npy_path.exists():
            npy_path = DATASET_ROOT / f"data/renji_npy_direct/test/{img_id}_1m.npy"
        if not npy_path.exists():
            npy_path = DATASET_ROOT / f"data/renji_npy_direct/test/{img_id}_2m.npy"
        shape = (256, 256)
        if npy_path.exists():
            arr = np.load(npy_path)
            shape = (arr.shape[0], arr.shape[1])

        result = roi_from_mask(mask_path, shape)
        if result is None:
            print(f"[WARN] Empty mask for {img_id}")
            continue
        center, scale = result
        roi_dict[img_id] = {"center": center.tolist(), "scale": scale}
        print(f"  {img_id}: center=({center[0]:.1f},{center[1]:.1f}), scale={scale:.3f}")

    # Generate visualizations
    print(f"\nGenerating visualizations...")
    for img_id, info in SELECTED_SAMPLES.items():
        if img_id not in roi_dict:
            print(f"  [SKIP] No ROI for {img_id}")
            continue

        roi = roi_dict[img_id]
        center = np.array(roi["center"], dtype=np.float32)
        scale = roi["scale"]

        npy_path = DATASET_ROOT / f"data/renji_npy_direct/test/{img_id}m.npy"
        if not npy_path.exists():
            npy_path = DATASET_ROOT / f"data/renji_npy_direct/test/{img_id}_1m.npy"
        if not npy_path.exists():
            npy_path = DATASET_ROOT / f"data/renji_npy_direct/test/{img_id}_2m.npy"

        if not npy_path.exists():
            print(f"  [SKIP] .npy not found for {img_id}")
            continue

        image_bgr = np.load(npy_path)
        if image_bgr.ndim == 2:
            image_bgr = np.repeat(image_bgr[:, :, None], 3, axis=2)

        label_path = npy_path.with_name(npy_path.name[:-5] + "l.npy")
        gt_pts = np.load(label_path).astype(np.float32)
        pred_pts = hrnet.predict(image_bgr, center, scale)
        resolution = get_resolution(res_map, img_id)

        errors = hungarian_error_mm(pred_pts, gt_pts, resolution)
        mean_err = float(errors.mean())

        category = info["category"]
        out_path = OUTPUT_VIS_DIR / f"{category}_{img_id}_vis.png"
        visualize_sample(image_bgr, pred_pts, gt_pts, center, scale, mean_err, category, out_path)
        print(f"  {img_id}: {mean_err:.2f}mm ({category}) -> {out_path.name}")

    print(f"\nDone! Saved to: {OUTPUT_VIS_DIR}")


if __name__ == "__main__":
    main()
