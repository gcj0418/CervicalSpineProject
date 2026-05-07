#!/usr/bin/env python

import argparse
import os
import csv

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import config.config as cfg
from data.load_test_cervical import TestData
from net.ceph_reg_refine_net import get_model
from utils import cal_acc, decode_reg


# Unified color palette (BGR, same as HRNet / VLD), extended to 14 colors
COLORS = [
    (195, 18, 251),
    (138, 209, 242),
    (20, 203, 39),
    (238, 229, 25),
    (187, 27, 251),
    (4, 183, 240),
    (138, 209, 115),
    (213, 249, 31),
    (185, 59, 76),
    (98, 242, 252),
    (255, 102, 0),
    (0, 128, 255),
    (128, 0, 128),
    (0, 200, 128),
]
GT_COLOR = (0, 255, 0)


def model_initial(model, model_name):
    pretrained_dict = torch.load(model_name)["model"]
    model_dict = model.state_dict()
    pretrained_dictf = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dictf)
    model.load_state_dict(model_dict)


def read_list_file(path):
    files = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                files.append(line)
    return files


def load_resolution_map(path):
    if not path:
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


def _ensure_uint8_rgb(image):
    """Ensure image is uint8 RGB/HWC suitable for cv2."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return image


def draw_compare(image, pred_coords, gt_coords):
    """Overlay GT (green) and Pred (per-landmark colors) on the same image, with index labels."""
    canvas = _ensure_uint8_rgb(image.copy())
    for idx, (x, y) in enumerate(gt_coords):
        ix, iy = int(round(x)), int(round(y))
        cv2.circle(canvas, (ix, iy), 4, GT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(canvas, str(idx + 1), (ix + 4, iy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, GT_COLOR, 1, cv2.LINE_AA)
    for idx, (x, y) in enumerate(pred_coords):
        ix, iy = int(round(x)), int(round(y))
        color = COLORS[idx % len(COLORS)]
        cv2.circle(canvas, (ix, iy), 3, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(idx + 1), (ix + 4, iy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return canvas


def draw_side_by_side(image, pred_coords, gt_coords):
    """GT on left, Pred on right, separated by a white 24px gap."""
    left = _ensure_uint8_rgb(image.copy())
    right = _ensure_uint8_rgb(image.copy())
    for x, y in gt_coords:
        ix, iy = int(round(x)), int(round(y))
        cv2.circle(left, (ix, iy), 3, GT_COLOR, -1, cv2.LINE_AA)
    for idx, (x, y) in enumerate(pred_coords):
        ix, iy = int(round(x)), int(round(y))
        color = COLORS[idx % len(COLORS)]
        cv2.circle(right, (ix, iy), 3, color, -1, cv2.LINE_AA)
    gap = np.full((image.shape[0], 24, 3), 255, dtype=np.uint8)
    return np.concatenate([left, gap, right], axis=1)


def draw_overlay(image, pred_coords, gt_coords):
    """Minimal backward-compatible overlay (single image). Kept for compatibility."""
    canvas = _ensure_uint8_rgb(image.copy())
    for idx, (x, y) in enumerate(gt_coords):
        ix, iy = int(round(x)), int(round(y))
        cv2.circle(canvas, (ix, iy), 3, GT_COLOR, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(idx + 1), (ix + 4, iy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, GT_COLOR, 1, cv2.LINE_AA)
    for idx, (x, y) in enumerate(pred_coords):
        ix, iy = int(round(x)), int(round(y))
        color = COLORS[idx % len(COLORS)]
        cv2.circle(canvas, (ix, iy), 3, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(idx + 1), (ix + 4, iy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return canvas


def main(args):
    cfg.PointNms = args.point_nms
    cfg.CLASS_NUMS = args.point_nms

    file_list = read_list_file(args.test_list)
    resolution_map = load_resolution_map(args.resolution_csv)
    test_loader = DataLoader(
        TestData(file_list, default_resolution=args.default_resolution, resolution_map=resolution_map),
        num_workers=0,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    vis_base_dir = args.vis_dir
    if vis_base_dir:
        os.makedirs(vis_base_dir, exist_ok=True)
        os.makedirs(os.path.join(vis_base_dir, "compare"), exist_ok=True)
        os.makedirs(os.path.join(vis_base_dir, "side_by_side"), exist_ok=True)

    model = get_model(num_layers=34, heads={"hm": 1, "class": cfg.PointNms}, NLayer1=2, NLayer2=4)
    model_initial(model, args.model)

    model.cuda()
    model.eval()

    total_counts = []
    sample_reports = []
    total_masks = 0
    for idx, (row_img, test_data, label_coords_, scalek, _, resol) in enumerate(test_loader):
        test_data = test_data.cuda().float()
        scalek = scalek.squeeze().numpy()
        resol = resol.squeeze().numpy()

        with torch.no_grad():
            outputs, _, _ = model(test_data)
            pred = outputs[-1][:, :, :2]
            key_points, mask_ = decode_reg(pred)
            counts, pred_coords = cal_acc(
                torch.squeeze(row_img).numpy(),
                key_points,
                mask_,
                label_coords_.squeeze().numpy(),
                scalek,
                resol,
            )
            total_counts.append(counts)
            sample_reports.append(
                {
                    "sample": os.path.basename(file_list[idx]).replace("m.npy", ""),
                    "mean_error": float(np.mean(counts)),
                    "min_error": float(np.min(counts)),
                    "max_error": float(np.max(counts)),
                    "resolution": "x".join([f"{v:.4f}" for v in np.asarray(resol).reshape(-1)]),
                }
            )
            total_masks += np.sum(mask_)

            if vis_base_dir:
                base_name = os.path.basename(file_list[idx]).replace("m.npy", "")
                raw_img = torch.squeeze(row_img).numpy()
                gt_coords = label_coords_.squeeze().numpy()

                compare = draw_compare(raw_img, pred_coords, gt_coords)
                side = draw_side_by_side(raw_img, pred_coords, gt_coords)

                cv2.imwrite(os.path.join(vis_base_dir, "compare", f"{base_name}_compare.png"), compare)
                cv2.imwrite(os.path.join(vis_base_dir, "side_by_side", f"{base_name}_side_by_side.png"), side)

            if args.save_preds:
                base_name = os.path.basename(file_list[idx]).replace("m.npy", "")
                pred_dir = os.path.join(args.output_dir, "predictions")
                os.makedirs(pred_dir, exist_ok=True)
                np.save(os.path.join(pred_dir, f"{base_name}_pred.npy"), pred_coords)
                np.save(os.path.join(pred_dir, f"{base_name}_gt.npy"), label_coords_.squeeze().numpy())

    total_counts = np.array(total_counts)
    num = len(total_counts)
    total_points = max(num * cfg.PointNms, 1)

    print(f"samples = {num}")
    print(f"total_points = {total_points}, total_masks = {int(total_masks)}")
    print(f"2mm acc = {np.sum(total_counts < cfg.ERROR_RANGE[0]) / total_points:.6f}")
    print(f"2.5mm acc = {np.sum(total_counts < cfg.ERROR_RANGE[1]) / total_points:.6f}")
    print(f"3mm acc = {np.sum(total_counts < cfg.ERROR_RANGE[2]) / total_points:.6f}")
    print(f"4mm acc = {np.sum(total_counts < cfg.ERROR_RANGE[3]) / total_points:.6f}")
    mean_error = float(np.mean(total_counts))
    print(f"mean error = {mean_error:.6f}")

    sample_reports = sorted(sample_reports, key=lambda x: x["mean_error"], reverse=True)
    ranking_path = os.path.join(args.output_dir, "sample_error_ranking.csv")
    with open(ranking_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "sample", "mean_error", "min_error", "max_error", "resolution"])
        writer.writeheader()
        for rank, item in enumerate(sample_reports, start=1):
            writer.writerow({"rank": rank, **item})

    # Write comparison_table.md (same schema as HRNet eval_spine.py)
    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    comparison_md = os.path.join(logs_dir, "comparison_table.md")
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(args.test_list)))
    with open(comparison_md, "w", encoding="utf-8") as f:
        f.write("| dataset | model | mean_error_mm | acc@2mm | acc@2.5mm | acc@3mm | acc@4mm |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        f.write(
            "| {dataset} | {model} | {mean:.4f} | {a2:.4f} | {a25:.4f} | {a3:.4f} | {a4:.4f} |\n".format(
                dataset=dataset_name if dataset_name else "unknown",
                model=os.path.basename(args.model),
                mean=mean_error,
                a2=np.sum(total_counts < cfg.ERROR_RANGE[0]) / total_points,
                a25=np.sum(total_counts < cfg.ERROR_RANGE[1]) / total_points,
                a3=np.sum(total_counts < cfg.ERROR_RANGE[2]) / total_points,
                a4=np.sum(total_counts < cfg.ERROR_RANGE[3]) / total_points,
            )
        )
    print(f"saved comparison table to {comparison_md}")

    worst_k = min(args.topk, len(sample_reports))
    print(f"top_{worst_k}_worst_samples:")
    for rank, item in enumerate(sample_reports[:worst_k], start=1):
        print(
            f"  {rank:02d}. {item['sample']} | mean_error={item['mean_error']:.4f} | "
            f"min={item['min_error']:.4f} | max={item['max_error']:.4f} | resolution={item['resolution']}"
        )
    print(f"ranking saved to {ranking_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test D-CeLR for cervical landmark regression")
    parser.add_argument("--test_list", type=str, required=True, help="Path to test.txt generated by converter")
    parser.add_argument("--model", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--point_nms", type=int, default=56)
    parser.add_argument("--default_resolution", type=float, default=1.0)
    parser.add_argument("--resolution_csv", type=str, default="", help="Optional CSV file mapping sample name to mm/pixel spacing")
    parser.add_argument("--vis_dir", type=str, default="", help="Directory to save overlay visualizations (compare/ + side_by_side/)")
    parser.add_argument("--save_preds", action="store_true", help="Save predicted coordinates to output_dir/predictions/")
    parser.add_argument("--topk", type=int, default=10, help="How many worst samples to print")
    parser.add_argument("--output_dir", type=str, default="outputs/cervical_eval", help="Directory to save ranking reports and comparison_table.md")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by current D-CeLR model implementation")

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
