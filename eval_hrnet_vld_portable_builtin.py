#!/usr/bin/env python3
"""
Evaluate hrnet-vld_portable with builtin nnU-Net ROI predictor.

This script:
  1. Converts RENJI test .npy images to temporary .nii.gz
  2. Runs nnU-Net ROI prediction to get masks
  3. Derives center/scale from each mask
  4. Runs HRNet inference with those ROIs
  5. Evaluates with Hungarian matching (unified metric)

Usage:
    conda activate dl_env
    python eval_hrnet_vld_portable_builtin.py
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
HRNET_VLD_ROOT = PROJECT_ROOT / "hrnet-vld_portable"
TEST_LIST = PROJECT_ROOT / "D-CeLR" / "data" / "renji_npy_direct" / "test.txt"
DATASET_ROOT = PROJECT_ROOT / "D-CeLR"
RESOLUTION_CSV = PROJECT_ROOT / "D-CeLR" / "data" / "renji_npy_direct" / "test_resolution.csv"

sys.path.insert(0, str(HRNET_VLD_ROOT))

# ---------------------------------------------------------------------------
# Imports from hrnet-vld_portable
# ---------------------------------------------------------------------------
from inference_with_roi import HRNetROIInference
from inference.roi_predictor import BuiltinROIPredictor

# scipy for Hungarian matching
try:
    from scipy.optimize import linear_sum_assignment
except Exception as exc:
    raise ImportError("scipy is required for Hungarian matching") from exc

# SimpleITK for .npy -> .nii.gz conversion
try:
    import SimpleITK as sitk
except Exception as exc:
    raise ImportError(
        "SimpleITK is required. Install it via: pip install SimpleITK"
    ) from exc


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


def roi_from_mask(mask_path: Path, fallback_shape: tuple[int, int]) -> tuple[np.ndarray, float] | None:
    """Derive center/scale from a binary mask (same logic as visualize_with_roi)."""
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


def npy_to_niigz(npy_path: Path, out_nii: Path) -> None:
    """Convert a .npy (H,W,C uint8) to .nii.gz for nnU-Net."""
    arr = np.load(npy_path)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Grayscale medical image: R==G==B, take first channel
        arr_2d = arr[:, :, 0]
    elif arr.ndim == 2:
        arr_2d = arr
    else:
        raise ValueError(f"Unexpected shape {arr.shape} for {npy_path}")

    if arr_2d.dtype != np.float32:
        arr_2d = arr_2d.astype(np.float32)

    # nnU-Net expects images as 3D (1, H, W) or 2D (H, W)
    # sitkImageFromArray expects (H, W) for 2D
    image = sitk.GetImageFromArray(arr_2d)
    sitk.WriteImage(image, str(out_nii))


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


def compute_metrics(errors_mm: np.ndarray, thresholds=(2.0, 2.5, 3.0, 4.0)) -> dict:
    errors = np.asarray(errors_mm, dtype=np.float32)
    mean_err = float(errors.mean()) if errors.size else 0.0
    median_err = float(np.median(errors)) if errors.size else 0.0
    std_err = float(errors.std()) if errors.size else 0.0
    accs = {thr: float((errors <= thr).mean()) if errors.size else 0.0 for thr in thresholds}
    return {"mean": mean_err, "median": median_err, "std": std_err, "accs": accs}


def main():
    print("=" * 60)
    print("Evaluating hrnet-vld_portable with builtin nnU-Net ROI")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load test samples
    # ------------------------------------------------------------------
    samples = []
    with TEST_LIST.open("r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            img_path = DATASET_ROOT / rel
            img_id = img_path.name[:-5]  # strip 'm.npy'
            label_path = img_path.with_name(img_path.name[:-5] + "l.npy")
            samples.append((img_id, img_path, label_path))

    print(f"Test samples: {len(samples)}")

    # ------------------------------------------------------------------
    # 2. Convert .npy -> temporary .nii.gz for nnU-Net
    # ------------------------------------------------------------------
    temp_nii_dir = Path(tempfile.mkdtemp(prefix="renji_nii_"))
    nii_paths: list[Path] = []
    for img_id, img_path, _ in samples:
        nii_out = temp_nii_dir / f"{img_id}.nii.gz"
        npy_to_niigz(img_path, nii_out)
        nii_paths.append(nii_out)
    print(f"Converted {len(nii_paths)} images to {temp_nii_dir}")

    # ------------------------------------------------------------------
    # 3. Initialize models
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hrnet_cfg = HRNET_VLD_ROOT / "experiments" / "renji" / "spine_renji_hrnet_w18_pretrained.yaml"
    hrnet_weights = HRNET_VLD_ROOT / "weights" / "model_best.pth"

    print(f"Loading HRNet: {hrnet_weights}")
    hrnet = HRNetROIInference(cfg_path=str(hrnet_cfg), model_path=str(hrnet_weights), device=device)

    print("Loading nnU-Net ROI predictor...")
    roi_predictor = BuiltinROIPredictor(device=device)

    # ------------------------------------------------------------------
    # 4. Predict ROI masks
    # ------------------------------------------------------------------
    mask_output_dir = temp_nii_dir / "roi_masks"
    print(f"Running nnU-Net ROI prediction -> {mask_output_dir}")
    try:
        mask_map = roi_predictor.predict_masks(nii_paths, mask_output_dir)
    except Exception as exc:
        print(f"ERROR: nnU-Net ROI prediction failed: {exc}")
        print("Make sure nnunetv2 is installed and nnUNetv2_predict is in PATH.")
        sys.exit(1)

    print(f"ROI masks generated: {len(mask_map)} / {len(nii_paths)}")

    # ------------------------------------------------------------------
    # 5. Run HRNet inference with ROI from masks
    # ------------------------------------------------------------------
    res_map = load_resolution_map(RESOLUTION_CSV)
    all_errors_mm = []

    for idx, (img_id, img_path, label_path) in enumerate(samples):
        # Load image for HRNet (needs BGR uint8)
        image = np.load(img_path)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).clip(0, 255)
            image = image.clip(0, 255).astype(np.uint8)

        # Get ROI from mask
        mask_path = mask_map.get(img_id)
        if mask_path is not None and mask_path.exists():
            roi = roi_from_mask(mask_path, image.shape[:2])
            if roi is not None:
                center, scale = roi
            else:
                print(f"  Warning: mask empty for {img_id}, falling back to image_stats")
                from inference.visualize_with_roi import image_stats_roi
                center, scale = image_stats_roi(image)
        else:
            print(f"  Warning: no mask for {img_id}, falling back to image_stats")
            from inference.visualize_with_roi import image_stats_roi
            center, scale = image_stats_roi(image)

        # Predict
        pred = hrnet.predict(image, center, scale)

        # Evaluate vs GT
        gt = np.load(label_path).astype(np.float32)
        if gt.shape[0] > 56:
            gt = gt[:56]

        resolution = get_resolution(res_map, img_id)
        errors = hungarian_error_mm(pred, gt, resolution)
        all_errors_mm.extend(errors.tolist())
        print(f"  {img_id}: mean_err={errors.mean():.3f}mm")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    all_errors_mm = np.array(all_errors_mm, dtype=np.float32)
    metrics = compute_metrics(all_errors_mm)

    print("\n" + "=" * 60)
    print("Results: hrnet-vld_portable + builtin nnU-Net ROI")
    print("=" * 60)
    print(f"Samples:    {len(samples)}")
    print(f"Landmarks:  {len(all_errors_mm)}")
    print(f"Mean:       {metrics['mean']:.4f} mm")
    print(f"Median:     {metrics['median']:.4f} mm")
    print(f"Std:        {metrics['std']:.4f} mm")
    print("-" * 60)
    for thr in sorted(metrics["accs"].keys()):
        print(f"Acc @ {thr}mm:  {metrics['accs'][thr]*100:.2f}%")
    print("=" * 60)

    # Cleanup temp dir (optional)
    # import shutil
    # shutil.rmtree(temp_nii_dir)
    print(f"\nTemp files kept at: {temp_nii_dir}")


if __name__ == "__main__":
    main()
