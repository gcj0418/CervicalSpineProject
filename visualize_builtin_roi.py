#!/usr/bin/env python3
"""
Visualize nnU-Net ROI + HRNet predictions on validation/test set.

Generates side-by-side comparison images with:
  - nnU-Net predicted ROI rectangle
  - Predicted keypoints (red)
  - GT keypoints (green)
  - Per-sample error annotation

Usage:
    export PATH="/c/Users/zzz/.conda/envs/dl_env/Scripts:$PATH"
    python visualize_builtin_roi.py
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
HRNET_VLD_ROOT = PROJECT_ROOT / "hrnet-vld_portable"
TEST_LIST = PROJECT_ROOT / "D-CeLR" / "data" / "renji_npy_direct" / "test.txt"
DATASET_ROOT = PROJECT_ROOT / "D-CeLR"
RESOLUTION_CSV = PROJECT_ROOT / "D-CeLR" / "data" / "renji_npy_direct" / "test_resolution.csv"
RENJI_RAW_ROOT = PROJECT_ROOT / "data" / "RENJI"
OUTPUT_VIS_DIR = PROJECT_ROOT / "outputs" / "builtin_roi_vis"

sys.path.insert(0, str(HRNET_VLD_ROOT))

from inference_with_roi import HRNetROIInference
from inference.roi_predictor import BuiltinROIPredictor
from visualization.draw import draw_prediction

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
    """Draw the crop rectangle used by HRNet (256x256 input size)."""
    canvas = image.copy()
    # HRNet crop size = scale * 200
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
    out_path: Path,
) -> None:
    """Create visualization with ROI rect, pred (red), GT (green)."""
    canvas = draw_prediction(image_bgr, pred_pts, show_boxes=True, show_keypoints=True, show_labels=True)

    # Draw GT keypoints in green
    for pt in gt_pts:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(canvas, (x, y), 3, (0, 255, 0), -1, cv2.LINE_AA)

    # Draw ROI rectangle
    canvas = draw_roi_rect(canvas, center, scale, color=(0, 255, 255), thickness=2)

    # Add error text
    h, w = canvas.shape[:2]
    text = f"mean_err={mean_err:.2f}mm"
    cv2.putText(canvas, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Visualizing nnU-Net ROI + HRNet predictions")
    print("=" * 60)

    OUTPUT_VIS_DIR.mkdir(parents=True, exist_ok=True)

    # Load samples
    samples = []
    with TEST_LIST.open("r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            img_path = DATASET_ROOT / rel
            img_id = img_path.name[:-5]
            label_path = img_path.with_name(img_path.name[:-5] + "l.npy")
            samples.append((img_id, img_path, label_path))

    print(f"Test samples: {len(samples)}")

    # Find original .nii.gz
    nii_map: dict[str, Path] = {}
    for img_id, _, _ in samples:
        nii_path = find_original_niigz(img_id)
        if nii_path is not None:
            nii_map[img_id] = nii_path

    print(f"Found original .nii.gz: {len(nii_map)}/{len(samples)}")

    # Init models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hrnet_cfg = HRNET_VLD_ROOT / "experiments" / "renji" / "spine_renji_hrnet_w18_pretrained.yaml"
    hrnet_weights = HRNET_VLD_ROOT / "weights" / "model_best.pth"
    print(f"Loading HRNet...")
    hrnet = HRNetROIInference(cfg_path=str(hrnet_cfg), model_path=str(hrnet_weights), device=device)
    print(f"Loading nnU-Net ROI predictor...")
    roi_predictor = BuiltinROIPredictor(device=device)

    res_map = load_resolution_map(RESOLUTION_CSV)

    # Process in batches
    temp_nii_dir = Path(tempfile.mkdtemp(prefix="renji_nii_vis_"))
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

    total = len(nii_paths)
    batch_size = 10
    roi_dict: dict[str, dict] = {}

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = nii_paths[start:end]
        batch_img_ids = [p.name[:-7] for p in batch]
        print(f"Processing batch {start//batch_size + 1}/{(total-1)//batch_size + 1} ({start}-{end})")

        mask_output_dir = temp_nii_dir / "roi_masks" / f"batch_{start}"
        mask_paths = roi_predictor.predict_masks(batch, mask_output_dir)

        for img_id in batch_img_ids:
            mask_path = mask_paths.get(img_id)
            if mask_path is None:
                print(f"  [WARN] No mask for {img_id}")
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
                print(f"  [WARN] Empty mask for {img_id}")
                continue
            center, scale = result
            roi_dict[img_id] = {
                "center": center.tolist(),
                "scale": scale,
            }
            print(f"  {img_id}: OK")

    print(f"ROI ready for {len(roi_dict)} samples, generating visualizations...")

    # Generate visualization for each sample
    summary_lines = ["# Builtin nnU-Net ROI Visualization Results\n", "| Sample | Mean Error (mm) | Status |\n", "|---|---|---|\n"]

    for img_id, img_path, label_path in samples:
        if img_id not in roi_dict:
            summary_lines.append(f"| {img_id} | - | No ROI |\n")
            continue

        roi = roi_dict[img_id]
        center = np.array(roi["center"], dtype=np.float32)
        scale = roi["scale"]

        image_bgr = np.load(img_path)
        if image_bgr.ndim == 2:
            image_bgr = np.repeat(image_bgr[:, :, None], 3, axis=2)

        gt_pts = np.load(label_path).astype(np.float32)
        pred_pts = hrnet.predict(image_bgr, center, scale)
        resolution = get_resolution(res_map, img_id)

        errors = hungarian_error_mm(pred_pts, gt_pts, resolution)
        mean_err = float(errors.mean())

        out_path = OUTPUT_VIS_DIR / f"{img_id}_vis.png"
        visualize_sample(image_bgr, pred_pts, gt_pts, center, scale, mean_err, out_path)

        status = "Good" if mean_err < 3.0 else ("OK" if mean_err < 5.0 else "Poor")
        summary_lines.append(f"| {img_id} | {mean_err:.2f} | {status} |\n")
        print(f"  {img_id}: {mean_err:.2f}mm -> {out_path.name}")

    # Save summary markdown
    summary_path = OUTPUT_VIS_DIR / "summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.writelines(summary_lines)

    print(f"\nDone! Visualizations saved to: {OUTPUT_VIS_DIR}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
