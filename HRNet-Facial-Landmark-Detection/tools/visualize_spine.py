#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.config import config, update_config
from lib.datasets import get_dataset


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


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize spine predictions')
    parser.add_argument('--cfg', required=True, type=str, help='experiment configuration filename')
    parser.add_argument('--prediction-file', required=True, type=str, help='torch predictions.pth file')
    parser.add_argument('--output-dir', required=True, type=str, help='directory to save visualizations')
    parser.add_argument('--max-samples', default=12, type=int, help='max number of samples to render')
    args = parser.parse_args()
    update_config(config, args)
    return args


def load_image(path: str) -> np.ndarray:
    image = np.load(path)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[2] == 3 else image


def draw_compare(image: np.ndarray, gt_pts: np.ndarray, pred_pts: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    for idx, pt in enumerate(gt_pts):
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, str(idx + 1), (int(pt[0]) + 4, int(pt[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    for idx, pt in enumerate(pred_pts):
        color = COLORS[idx % len(COLORS)]
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), 3, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(idx + 1), (int(pt[0]) + 4, int(pt[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return canvas


def draw_side_by_side(image: np.ndarray, gt_pts: np.ndarray, pred_pts: np.ndarray) -> np.ndarray:
    left = image.copy()
    right = image.copy()
    for pt in gt_pts:
        cv2.circle(left, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1, cv2.LINE_AA)
    for idx, pt in enumerate(pred_pts):
        color = COLORS[idx % len(COLORS)]
        cv2.circle(right, (int(pt[0]), int(pt[1])), 3, color, -1, cv2.LINE_AA)
    gap = np.full((image.shape[0], 24, 3), 255, dtype=np.uint8)
    return np.concatenate([left, gap, right], axis=1)


def main():
    args = parse_args()
    dataset_type = get_dataset(config)
    dataset = dataset_type(config, is_train=False)
    predictions = torch.load(args.prediction_file, map_location='cpu')

    output_dir = Path(args.output_dir)
    compare_dir = output_dir / 'compare'
    side_dir = output_dir / 'side_by_side'
    compare_dir.mkdir(parents=True, exist_ok=True)
    side_dir.mkdir(parents=True, exist_ok=True)

    count = min(len(dataset), len(predictions), args.max_samples)
    for idx in range(count):
        _, _, meta = dataset[idx]
        image = load_image(meta['image_path'])
        image = to_bgr(image)
        gt_pts = meta['pts'].cpu().numpy()
        pred_pts = predictions[idx].cpu().numpy()

        compare = draw_compare(image, gt_pts, pred_pts)
        side = draw_side_by_side(image, gt_pts, pred_pts)

        stem = f"{idx:03d}_{meta['img_id']}"
        cv2.imwrite(str(compare_dir / f'{stem}.png'), compare)
        cv2.imwrite(str(side_dir / f'{stem}.png'), side)


if __name__ == '__main__':
    main()