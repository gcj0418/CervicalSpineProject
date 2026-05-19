#!/usr/bin/env python3
"""
Evaluate hrnet-vld_portable with builtin nnU-Net ROI predictor (v2).
Uses original .nii.gz from data/RENJI/ instead of converted .npy.

This script:
  1. Finds original .nii.gz files in data/RENJI/ matching test set img_ids
  2. Runs nnU-Net ROI prediction on original images
  3. Derives center/scale from each mask
  4. Runs HRNet inference with those ROIs (using .npy images)
  5. Evaluates with Hungarian matching (unified metric)

Usage:
    export PATH="/c/Users/zzz/.conda/envs/dl_env/Scripts:$PATH"
    python eval_hrnet_vld_portable_builtin_v2.py
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import time
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
RENJI_RAW_ROOT = PROJECT_ROOT / "data" / "RENJI"

sys.path.insert(0, str(HRNET_VLD_ROOT))

from inference_with_roi import HRNetROIInference
from inference.roi_predictor import BuiltinROIPredictor

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise ImportError("scipy is required for Hungarian matching") from exc

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("SimpleITK is required") from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
    """Find the original .nii.gz in data/RENJI/ matching img_id."""
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


def compute_metrics(errors_mm: np.ndarray, thresholds=(2.0, 2.5, 3.0, 4.0)) -> dict:
    errors = np.asarray(errors_mm, dtype=np.float32)
    mean_err = float(errors.mean()) if errors.size else 0.0
    median_err = float(np.median(errors)) if errors.size else 0.0
    std_err = float(errors.std()) if errors.size else 0.0
    accs = {thr: float((errors <= thr).mean()) if errors.size else 0.0 for thr in thresholds}
    return {"mean": mean_err, "median": median_err, "std": std_err, "accs": accs}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Evaluating hrnet-vld_portable with builtin nnU-Net ROI (v2)")
    print("Using original .nii.gz from data/RENJI/")
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
    # 2. Find original .nii.gz files
    # ------------------------------------------------------------------
    nii_map: dict[str, Path] = {}
    missing = []
    for img_id, _, _ in samples:
        nii_path = find_original_niigz(img_id)
        if nii_path is None:
            # Try without suffix variations
            base_id = img_id.rsplit("_", 1)[0] if "_" in img_id else img_id
            nii_path = find_original_niigz(base_id)
        if nii_path is not None:
            nii_map[img_id] = nii_path
        else:
            missing.append(img_id)

    print(f"Found original .nii.gz: {len(nii_map)}/{len(samples)}")
    if missing:
        print(f"Missing: {missing}")

    if len(nii_map) == 0:
        print("No original .nii.gz found, aborting.")
        return

    # ------------------------------------------------------------------
    # 3. Copy/link .nii.gz to temp dir for nnU-Net
    # ------------------------------------------------------------------
    temp_nii_dir = Path(tempfile.mkdtemp(prefix="renji_nii_v2_"))
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
    print(f"Prepared {len(nii_paths)} images in {temp_nii_dir}")

    # ------------------------------------------------------------------
    # 4. Initialize models
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    hrnet_cfg = HRNET_VLD_ROOT / "experiments" / "renji" / "spine_renji_hrnet_w18_pretrained.yaml"
    hrnet_weights = HRNET_VLD_ROOT / "weights" / "model_best.pth"

    print(f"Loading HRNet: {hrnet_weights}")
    hrnet = HRNetROIInference(cfg_path=str(hrnet_cfg), model_path=str(hrnet_weights), device=device)

    print("Loading nnU-Net ROI predictor...")
    roi_predictor = BuiltinROIPredictor(device=device)

    # ------------------------------------------------------------------
    # 5. Predict ROI masks (model resident in memory)
    # ------------------------------------------------------------------
    mask_output_dir = temp_nii_dir / "roi_masks"
    mask_output_dir.mkdir(exist_ok=True)

    roi_dict: dict[str, dict] = {}
    total = len(nii_paths)

    print(f"Predicting ROI masks for all {total} images (model already loaded)...")
    t0 = time.time()
    mask_paths = roi_predictor.predict_masks(nii_paths, mask_output_dir)
    t1 = time.time()
    print(f"ROI prediction done in {t1 - t0:.1f}s ({(t1 - t0) / total:.2f}s per image)")

    for nii_path in nii_paths:
        img_id = nii_path.name[:-7] if nii_path.name.endswith('.nii.gz') else nii_path.stem
        mask_path = mask_paths.get(img_id)
        if mask_path is None:
            print(f"  [WARN] No mask for {img_id}")
            continue
        # Get shape from corresponding .npy
        npy_path = DATASET_ROOT / f"data/renji_npy_direct/test/{img_id}m.npy"
        if not npy_path.exists():
            npy_path = DATASET_ROOT / f"data/renji_npy_direct/test/{img_id}_1m.npy"
        if not npy_path.exists():
            npy_path = DATASET_ROOT / f"data/renji_npy_direct/test/{img_id}_2m.npy"
        if npy_path.exists():
            arr = np.load(npy_path)
            shape = (arr.shape[0], arr.shape[1])
        else:
            shape = (256, 256)
            print(f"  [WARN] Could not find .npy for {img_id}, using fallback shape")

        result = roi_from_mask(mask_path, shape)
        if result is None:
            print(f"  [WARN] Empty mask for {img_id}")
            continue
        center, scale = result
        roi_dict[img_id] = {
            "center": center.tolist(),
            "scale": scale,
        }
        print(f"  {img_id}: center=({center[0]:.1f},{center[1]:.1f}), scale={scale:.3f}")

    print(f"ROI masks generated: {len(roi_dict)} / {len(samples)}")

    # ------------------------------------------------------------------
    # 6. Run HRNet inference with nnU-Net ROIs
    # ------------------------------------------------------------------
    res_map = load_resolution_map(RESOLUTION_CSV)
    all_errors: list[float] = []

    for img_id, img_path, label_path in samples:
        if img_id not in roi_dict:
            print(f"[SKIP] No ROI for {img_id}")
            continue

        roi = roi_dict[img_id]
        center = np.array(roi["center"], dtype=np.float32)
        scale = roi["scale"]

        image_bgr = np.load(img_path)
        if image_bgr.ndim == 2:
            image_bgr = np.repeat(image_bgr[:, :, None], 3, axis=2)

        preds = hrnet.predict(image_bgr, center, scale)

        gt_pts = np.load(label_path).astype(np.float32)
        resolution = get_resolution(res_map, img_id)

        errors = hungarian_error_mm(preds, gt_pts, resolution)
        all_errors.extend(errors.tolist())

        sample_mean = float(errors.mean())
        print(f"  {img_id}: mean_err={sample_mean:.3f}mm")

    # ------------------------------------------------------------------
    # 7. Compute metrics
    # ------------------------------------------------------------------
    errors_np = np.array(all_errors, dtype=np.float32)
    metrics = compute_metrics(errors_np)

    print("\n" + "=" * 60)
    print("Results: hrnet-vld_portable + builtin nnU-Net ROI (v2)")
    print("=" * 60)
    print(f"Samples:    {len([s for s in samples if s[0] in roi_dict])}")
    print(f"Landmarks:  {len(all_errors)}")
    print(f"Mean:       {metrics['mean']:.4f} mm")
    print(f"Median:     {metrics['median']:.4f} mm")
    print(f"Std:        {metrics['std']:.4f} mm")
    print("-" * 60)
    for thr, acc in metrics["accs"].items():
        print(f"Acc @ {thr}mm:  {acc * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
