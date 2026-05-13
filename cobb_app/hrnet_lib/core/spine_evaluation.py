from __future__ import annotations

import csv
import os

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def load_resolution_map(path):
    if not path:
        return {}
    resolution_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) < 3:
                continue
            name = row[0].strip()
            try:
                resolution_map[name] = np.array([float(row[1]), float(row[2])], dtype=np.float32)
            except ValueError:
                continue
    return resolution_map


def get_sample_resolution(resolution_map, img_id, default_resolution):
    sample_key = os.path.splitext(img_id)[0]
    if img_id in resolution_map:
        return resolution_map[img_id]
    if sample_key in resolution_map:
        return resolution_map[sample_key]
    if '_' in sample_key:
        base_key = sample_key.rsplit('_', 1)[0]
        if base_key in resolution_map:
            return resolution_map[base_key]
    return default_resolution


def point_error_mm(pred_point, gt_point, resolution):
    diff = np.asarray(pred_point, dtype=np.float32) - np.asarray(gt_point, dtype=np.float32)
    resov_arr = np.asarray(resolution, dtype=np.float32).reshape(-1)
    if resov_arr.size >= 2:
        return np.sqrt((diff[0] * resov_arr[0]) ** 2 + (diff[1] * resov_arr[1]) ** 2)
    return np.sqrt(np.sum(np.power(diff, 2))) * float(resov_arr[0] if resov_arr.size else 1.0)


def pair_landmarks(pr_landmarks, gt_landmarks, resolution, matching_mode='hungarian'):
    pr_landmarks = np.asarray(pr_landmarks, dtype=np.float32)
    gt_landmarks = np.asarray(gt_landmarks, dtype=np.float32)
    if len(pr_landmarks) == 0 or len(gt_landmarks) == 0:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty, np.asarray([], dtype=np.float32)

    if matching_mode == 'hungarian':
        if linear_sum_assignment is None:
            raise ImportError('matching_mode=hungarian requires scipy (pip install scipy)')

        resov_arr = np.asarray(resolution, dtype=np.float32).reshape(-1)
        if resov_arr.size >= 2:
            spacing = np.array([resov_arr[0], resov_arr[1]], dtype=np.float32)
        else:
            scale = float(resov_arr[0] if resov_arr.size else 1.0)
            spacing = np.array([scale, scale], dtype=np.float32)

        pr_scaled = pr_landmarks * spacing
        gt_scaled = gt_landmarks * spacing
        cost = np.linalg.norm(pr_scaled[:, None, :] - gt_scaled[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        pr_matched = pr_landmarks[row_ind]
        gt_matched = gt_landmarks[col_ind]
        dists = cost[row_ind, col_ind].astype(np.float32)
        return pr_matched, gt_matched, dists

    raise ValueError('Only matching_mode=hungarian is supported, got {}'.format(matching_mode))
