#!/usr/bin/env python3
"""
Ensemble spine landmark predictions from VLD / D-CeLR / HRNet.
Supports: mean, median, weighted average fusion.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def load_resolution_map(path):
    if not path or not Path(path).exists():
        return {}
    resolution_map = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            name = row[0].strip()
            try:
                if len(row) >= 3:
                    resolution_map[name] = np.array([float(row[1]), float(row[2])], dtype=np.float32)
                else:
                    resolution_map[name] = float(row[1])
            except ValueError:
                continue
    return resolution_map


def get_sample_resolution(resolution_map, sample_key, default=1.0):
    sample_key = os.path.splitext(sample_key)[0]
    if sample_key in resolution_map:
        return resolution_map[sample_key]
    return default


def load_hrnet_preds(pred_file):
    """HRNet: predictions.pth -> tensor [N, num_joints, 2]"""
    preds = torch.load(pred_file, map_location="cpu", weights_only=False)
    if isinstance(preds, dict):
        # Some HRNet versions save dict
        return preds
    # tensor format: need dataset to map index -> img_id
    return preds


def load_vld_preds(pred_file):
    """VLD: predictions.pth -> dict {img_id: ndarray [num_joints, 2]}"""
    preds = torch.load(pred_file, map_location="cpu", weights_only=False)
    # Convert torch tensors to numpy arrays
    return {k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v, dtype=np.float32)) for k, v in preds.items()}


def load_dcelr_preds(pred_dir):
    """D-CeLR: {name}_pred.npy -> dict {name: ndarray [num_joints, 2]}"""
    pred_dir = Path(pred_dir)
    preds = {}
    for p in sorted(pred_dir.glob("*_pred.npy")):
        name = p.name.replace("_pred.npy", "")
        preds[name] = np.load(p)
    return preds


def point_error_mm(pred_pts, gt_pts, resolution):
    """Compute per-landmark mm error after Hungarian matching pred to gt."""
    pred_pts = np.asarray(pred_pts, dtype=np.float32)
    gt_pts = np.asarray(gt_pts, dtype=np.float32)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.zeros(len(pred_pts), dtype=np.float32)

    # Hungarian matching: align pred to gt
    cost = np.linalg.norm(pred_pts[:, None, :] - gt_pts[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_pred = np.zeros_like(pred_pts)
    for r, c in zip(row_ind, col_ind):
        matched_pred[c] = pred_pts[r]

    diff = matched_pred - gt_pts
    res_arr = np.asarray(resolution, dtype=np.float32).reshape(-1)
    if res_arr.size >= 2:
        return np.sqrt((diff[:, 0] * res_arr[0]) ** 2 + (diff[:, 1] * res_arr[1]) ** 2)
    return np.sqrt(np.sum(diff ** 2, axis=1)) * float(res_arr[0] if res_arr.size else 1.0)


def align_landmarks_hungarian(pred_pts, ref_pts):
    """Reorder pred_pts to best match ref_pts using Hungarian matching on Euclidean distance.
    Returns pred_pts reordered so that aligned[i] corresponds to ref_pts[i]."""
    pred_pts = np.asarray(pred_pts, dtype=np.float32)
    ref_pts = np.asarray(ref_pts, dtype=np.float32)
    if len(pred_pts) == 0 or len(ref_pts) == 0:
        return pred_pts
    # cost[i, j] = distance between pred_pts[i] and ref_pts[j]
    cost = np.linalg.norm(pred_pts[:, None, :] - ref_pts[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    # row_ind[k] matched to col_ind[k]; we want output[j] = pred that matched ref[j]
    aligned = np.zeros_like(pred_pts)
    for r, c in zip(row_ind, col_ind):
        aligned[c] = pred_pts[r]
    return aligned


def compute_metrics(errors_mm):
    errors = np.asarray(errors_mm, dtype=np.float32)
    mean_err = float(errors.mean()) if errors.size else 0.0
    thresholds = [2.0, 2.5, 3.0, 4.0]
    accs = {thr: float((errors <= thr).mean()) if errors.size else 0.0 for thr in thresholds}
    return mean_err, accs


def normalize_key(k):
    """Strip file extension for cross-method matching."""
    return os.path.splitext(str(k))[0]


def align_samples(method_preds, method_names):
    """Find common sample keys across all methods (ignoring extensions)."""
    # Normalize all keys
    normalized_preds = []
    for preds in method_preds:
        norm = {}
        for k, v in preds.items():
            norm[normalize_key(k)] = v
        normalized_preds.append(norm)

    all_keys = [set(p.keys()) for p in normalized_preds]
    common = all_keys[0]
    for keys in all_keys[1:]:
        common &= keys
    print(f"Common samples: {len(common)} / {[len(k) for k in all_keys]} (by method)")
    missing = []
    for i, keys in enumerate(all_keys):
        diff = common ^ keys
        if diff:
            missing.append(f"{method_names[i]} missing/extra: {sorted(diff)[:5]}")
    if missing:
        for m in missing:
            print(f"  {m}")
    return normalized_preds, sorted(common)


def fuse_predictions(pred_list, weights=None, mode="mean"):
    """
    pred_list: list of ndarray [num_joints, 2]
    weights: list of float, same length as pred_list
    mode: 'mean', 'median', 'weighted'
    """
    preds = np.stack(pred_list, axis=0)  # [M, N, 2]
    if mode == "median":
        return np.median(preds, axis=0)
    if mode == "weighted" and weights is not None:
        weights = np.asarray(weights, dtype=np.float32)
        weights = weights / weights.sum()
        return np.average(preds, axis=0, weights=weights)
    return preds.mean(axis=0)


def main():
    parser = argparse.ArgumentParser(description="Ensemble spine landmark predictions")
    parser.add_argument("--hrnet_preds", type=str, default="", help="HRNet predictions.pth or predictions dir")
    parser.add_argument("--vld_preds", type=str, default="", help="VLD predictions.pth")
    parser.add_argument("--dcelr_preds", type=str, default="", help="D-CeLR predictions dir with *_pred.npy")
    parser.add_argument("--gt_source", type=str, default="", help="D-CeLR predictions dir with *_gt.npy (used as GT source)")
    parser.add_argument("--resolution_csv", type=str, default="", help="Spacing CSV")
    parser.add_argument("--default_resolution", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="outputs/ensemble")
    parser.add_argument("--fusion", type=str, default="mean,median,weighted", help="Fusion modes to try")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    resolution_map = load_resolution_map(args.resolution_csv)

    method_preds = []
    method_names = []

    # Load HRNet
    if args.hrnet_preds:
        hrnet_file = Path(args.hrnet_preds)
        if hrnet_file.is_dir():
            hrnet_file = hrnet_file / "predictions" / "predictions.pth"
        hrnet = load_hrnet_preds(str(hrnet_file))
        if isinstance(hrnet, dict):
            method_preds.append(hrnet)
        else:
            # tensor format: need img_id mapping from dataset (not available here)
            print("Warning: HRNet tensor format requires dataset mapping. Please use dict format.")
        method_names.append("HRNet")
        print(f"Loaded HRNet: {len(method_preds[-1])} samples")

    # Load VLD
    if args.vld_preds:
        vld_file = Path(args.vld_preds)
        if vld_file.is_dir():
            vld_file = vld_file / "predictions" / "predictions.pth"
        vld = load_vld_preds(str(vld_file))
        method_preds.append(vld)
        method_names.append("VLD")
        print(f"Loaded VLD: {len(method_preds[-1])} samples")

    # Load D-CeLR
    if args.dcelr_preds:
        dcelr = load_dcelr_preds(args.dcelr_preds)
        method_preds.append(dcelr)
        method_names.append("D-CeLR")
        print(f"Loaded D-CeLR: {len(method_preds[-1])} samples")

    if len(method_preds) < 2:
        raise ValueError("Need at least 2 methods for ensemble")

    # Align samples
    method_preds, common_samples = align_samples(method_preds, method_names)
    if not common_samples:
        raise ValueError("No common samples found across methods")

    # Load GT from D-CeLR (most reliable)
    gt_dict = {}
    if args.gt_source:
        gt_dir = Path(args.gt_source)
        for p in sorted(gt_dir.glob("*_gt.npy")):
            name = normalize_key(p.name.replace("_gt.npy", ""))
            if name in common_samples:
                gt_dict[name] = np.load(p)

    if not gt_dict:
        # Fallback: use one method's prediction as reference shape, but we need GT for metrics
        print("Warning: No GT provided, computing inter-method consistency only")

    # Fusion modes
    modes = [m.strip() for m in args.fusion.split(",")]

    # Validation weights (can be overridden; here use hardcoded e60 accuracies as proxy)
    # RUIJIN e60: VLD=0.710, HRNet=0.466, D-CeLR=0.293
    # RENJI e60: VLD=0.640, HRNet=0.327, D-CeLR=0.564
    weights_map = {
        "VLD": 0.50,
        "HRNet": 0.30,
        "D-CeLR": 0.20,
    }
    weights = [weights_map.get(n, 1.0) for n in method_names]
    print(f"Ensemble weights: {dict(zip(method_names, weights))}")

    results = []
    for mode in modes:
        all_errors = []
        for sample in common_samples:
            # Pick reference method (prefer VLD if available, else first)
            ref_idx = method_names.index("VLD") if "VLD" in method_names else 0
            ref_pts = method_preds[ref_idx][sample]

            # Align all methods to reference via Hungarian matching
            aligned_preds = []
            for i in range(len(method_preds)):
                pts = method_preds[i][sample]
                if i == ref_idx:
                    aligned_preds.append(pts)
                else:
                    aligned_preds.append(align_landmarks_hungarian(pts, ref_pts))

            fused = fuse_predictions(aligned_preds, weights=weights if mode == "weighted" else None, mode=mode)

            if sample in gt_dict:
                gt = gt_dict[sample]
                sample_key = sample
                res = get_sample_resolution(resolution_map, sample_key, args.default_resolution)
                errors = point_error_mm(fused, gt, res)
                all_errors.extend(errors.tolist())

        if all_errors:
            mean_err, accs = compute_metrics(all_errors)
            results.append({
                "mode": mode,
                "mean_error": mean_err,
                "acc@2": accs[2.0],
                "acc@2.5": accs[2.5],
                "acc@3": accs[3.0],
                "acc@4": accs[4.0],
            })
            print(
                f"[{mode:10s}] mean={mean_err:.4f}  acc@2={accs[2.0]:.4f}  acc@2.5={accs[2.5]:.4f}  "
                f"acc@3={accs[3.0]:.4f}  acc@4={accs[4.0]:.4f}"
            )

    # Write comparison table
    md_path = Path(args.output_dir) / "comparison_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| mode | mean_error_mm | acc@2mm | acc@2.5mm | acc@3mm | acc@4mm |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for r in results:
            f.write(
                f"| {r['mode']} | {r['mean_error']:.4f} | {r['acc@2']:.4f} | "
                f"{r['acc@2.5']:.4f} | {r['acc@3']:.4f} | {r['acc@4']:.4f} |\n"
            )
    print(f"Saved comparison table to {md_path}")


if __name__ == "__main__":
    main()
